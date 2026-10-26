# Manifest API and YAML fields

Compiled Research is authored in YAML and executed through the CLI. The [book chapter](Compiled-Research.md) develops the complete declaration; the [Compiled project](examples/compiled-research/README.md) supplies its exact executable source. The [Python project](examples/python-research/README.md) owns ordinary Python research functions.

This appendix documents YAML fields and the Manifest component APIs that implement them. Builder signatures and direct component calls are implementation and optional integration reference. They are not steps in the Compiled research workflow.

## Prerequisites

- `pip install "vaquum-limen[data]"` for the bundled data-backed templates
- a YAML file created with `limen init`; component authors may consult the optional builder APIs below
- the optional model extra required by the selected reference architecture

## Run the recorded YAML project

<!-- example: manifest-command source=docs/examples/commands.json region=compiled-yaml scenario=selection interface=compiled role=executed -->
```bash
limen run --no-progress-bar compiled-research/selection.yaml
```

The run writes a result directory containing the copied YAML manifest, `metadata.json`, `results.csv`, and `round_data.jsonl.gz`.

## YAML Shape

Mappings used as YAML keys are rejected during parsing, with the key's line and
column in the parse error. Manifest fields use named keys; nested mappings remain
available as values.

Every `{name}` used as a round-parameter lookup must have a candidate list in
`sfd.params`, including lookups in inactive transforms. Architecture defaults and
implicit controls such as `feature_groups` or `auto_pca` do not declare a lookup.
For example, an indicator period of `{C}` requires `sfd.params.C`; declaring it
also overrides the model's `C` setting. Use a distinct name when those settings
should vary independently. Missing declarations fail validation at the consumer's
field before data loading or execution. A declared axis with no consuming binding
fails at `sfd.params.<name>`, including singleton axes. The error reports its
declared choice count. Remove it or supply a real binding; traversal randomness
belongs at `uel.search_strategy.seed`.

Unknown and `**kwargs` architecture signatures, whole-parameter data extensions
and allocation context consumers remain indeterminate unless their complete
explicit reuse contracts declare finite dependencies. This suppresses only the
reverse unused-axis check; declaration and Boolean `include_if` checks still apply.
Real model arguments, native exact-name and formatted lookups, nested execution
and funding references, and applicable meta-controls count as consumers.

Saved CLI resume applies the same narrow unused-axis refusal after schema,
callback and structural admission, before checkpoint loading or reconstruction.
Assessment also refuses dead axes. Correct a legacy declaration and run it under
a new identity; recorded YAML is retained unchanged. Native caller-owned SFDs,
prevalidated compilation and scientific read-back keep their existing boundaries.

Model-source paths, transform group names and build-time `params_override`
values retain their literal meaning, including in family evidence. Separate
execution and funding source parameters use exact `{name}` lookups, including
complete keys such as `{file-path}`, and require declared candidates. Native
formatted bindings preserve escaped brace pairs; a live inner field such
as `roc_{{{period}}}` still requires `period`. Native transform parameters preserve
exact parameter-name lookup precedence and underscore-prefixed literals. Direct Python
Manifest/UEL calls may still supply values through `context_params`; a declared
YAML experiment must be complete without that external context. The existing
native formatted identifier-reference grammar is unchanged.

This is the Compiled project's complete selection definition. From the unpacked recorded-input bundle root, the command above writes `results/dev/compiled-cli`; the YAML supplies every execution choice:

<!-- example: manifest-selection source=docs/examples/compiled-research/selection.yaml region=__file__ scenario=selection interface=compiled role=executed -->
```yaml
# region declaration
schema_version: "1.0"
metadata:
  name: book_compiled_selection
  mode: development
sfd:
  manifest:
    type: ml
# region recorded_source
    data_source:
      method: limen.data.HistoricalData.get_any_file
      params:
        file_path_or_url: inputs/spot_15m_20250101_20250531.parquet
# endregion recorded_source
    split_dates:
      train_start: "2025-01-01"
      train_end: "2025-01-15"
      val_start: "2025-01-15"
      val_end: "2025-01-18"
      test_start: "2025-01-18"
      test_end: "2025-01-22"
# region momentum_indicator
    indicators:
      - func: limen.indicators.roc
        params:
          period: 1
          group: momentum
# endregion momentum_indicator
# region fitted_target
    target:
      name: quantile_flag
      class: limen.targets.QuantileBinaryTarget
      fit_params:
        source_column: roc_1
        quantile: 0.5
      transform_params:
        shift: -1
# endregion fitted_target
# region validation_objective
    objective:
      metric: backtest_total_return
      direction: maximize
# endregion validation_objective
    backtest:
      source_interval_seconds: 900
      fee_bps: 1.0
      slip_bps: 0.0
      product:
        kind: cash_spot
        instrument: BTCUSDT
        base_currency: BTC
        quote_currency: USDT
        quantity_step: 1.0e-9
        min_notional: 0.0
# region fitted_scaler
    scaler:
      from_params: scaler_type
# endregion fitted_scaler
    reference_architecture: limen.sfd.reference_architecture.logreg_binary
  params:
    scaler_type: [logreg]
    C: [1.0, 0.5]
    random_state: [42]
uel:
  output_path: compiled-cli
  n_permutations: 2
  search_strategy:
    type: grid
  output_format: csv

# region assessment
assessment:
  metric: val_backtest_total_return
  direction: maximize
  top_k: 1
  predict: true
# endregion assessment
# endregion declaration
```

`schema_version` is the internal Limen YAML contract marker, not a published JSON Schema or editor schema.
Only the exact string `"1.0"` is supported. Missing, null, non-string and unequal values fail at
`schema_version` before current-schema fields are interpreted. Values are not
trimmed, coerced or compared as package release versions. No older supported
schema exists, so there is no migration transform. `metadata.limen_version`
records a package release independently; an older compatible release remains
admissible.

This policy applies to programmatic `validate` and CLI validation, profile,
commit, reserve, assess, dry-run and run, including stored `manifest://` documents
and the saved YAML used by CLI resume. An unsupported saved schema is refused
before reconstruction or artifact changes. Direct `CompiledSFD` and
`build_manifest` calls require prevalidated input; they do not rerun validation.
Trainer/Sensor and other scientific read-back paths retain that reconstruction
boundary. This rule does not change unrelated integer schema versions in saved
reports or checkpoints.

Optional author metadata fields may be omitted; explicit `null` also means absence. Non-null values
follow these contracts; validation rejects malformed shapes at the field or tag
item before fresh CLI compilation or artifact publication.

| Field | Admitted value |
| --- | --- |
| `metadata.author` | Nonblank string. |
| `metadata.description` | String, including an empty string. |
| `metadata.tags` | List of nonblank strings. Empty lists and duplicates are allowed; names and order are preserved. |
| `metadata.created_at` | Quoted real calendar date `YYYY-MM-DD`, or `YYYY-MM-DDTHH:MM:SS` with optional 1–6 fractional digits and optional `Z` or `±HH:MM` offset. |
| `metadata.limen_version` | Quoted [Python package-version string](https://packaging.python.org/en/latest/specifications/version-specifiers/#appendix-parsing-version-strings-with-regular-expressions), including release, pre/post/dev and local forms and `0+unknown`. No whitespace padding or installed-version comparison. |

Quote author-created dates and timestamps, for example `created_at: "2025-10-25"`.
Unquoted YAML dates/timestamps are native objects that cannot enter the run's JSON
reference unchanged; validation rejects them with a quoting hint. Validation
preserves every declaration value. The author's optional timestamp is independent
of the runtime timestamp written to `metadata.json`; its timezone may be omitted.
Old and future package-version metadata do not establish schema compatibility.

These shapes apply to programmatic validation and fresh CLI admission, including
stored URI execution. Committed content identities are unchanged. Direct
prevalidated compilation and scientific reconstruction retain their existing
boundary.

`uel.n_permutations` is a positive integer execution budget. YAML validation rejects bool, zero, negative, and over-budget values before execution.

YAML anchors and aliases retain their logical type: Boolean values are rejected for `uel.n_permutations` and `uel.workers`, while positive integer spellings and anchored/aliased integers remain supported.

### The `uel` Execution Block

The `uel` block configures how the experiment runs. `n_permutations` is required; the rest are optional and fall back to the defaults below.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `n_permutations` | int | required | round budget; rejected if bool, zero, negative, or larger than the parameter space |
| `workers` | int | `1` | maximum concurrent isolated round evaluations; positive non-boolean integer; see [parallel rounds](#parallel-rounds) |
| `search_strategy.type` | `random`, `grid` or `sobol` | `random` | random sampling, exhaustive grid traversal or [seeded Sobol sampling](Advanced-Search.md#seeded-sobol-sampling) |
| `search_strategy.seed` | int | none | candidate-generation seed; required, non-boolean and nonnegative for Sobol |
| `search_strategy.shuffle` | bool | `false` | grid-only seeded finite permutation; true requires an integer `seed`; see [seeded traversal](Advanced-Search.md#seeded-finite-traversal) |
| `prep_each_round` | bool | `true` | compatibility setting; ordinary CLI/MSQ execution always prepares each candidate |
| `checkpoint_interval` | int | `1000` | rounds between checkpoint writes; positive non-boolean integer |
| `record_execution` | bool | `false` | record test snapshot series and unscaled market returns; see [UEL recording](Universal-Experiment-Loop.md#record-execution) |
| `record_model_outputs` | bool | `false` | record test probabilities and boosting iteration counts; see [UEL recording](Universal-Experiment-Loop.md#record-model-outputs) |
| `recording` | mapping of bool | all groups `false` | choose saved predictions, alignment arrays, event ledger and fold details; resume requires applicable groups; see [UEL recording policy](Universal-Experiment-Loop.md#recording-policy) |
| `factorize` | bool | `false` | reuse deterministic snapshot signal work across independent backtest-only grid axes; see [UEL](Universal-Experiment-Loop.md) |
| `feedback_interval` | int | `100` | rounds between feedback-controller triggers; positive non-boolean integer |
| `pruning_strategies` | list | none | reducers that run during feedback cycles; each item has `type` (a reducer key) and optional `params` |
| `output_format` | `csv` or `parquet` | `csv` | `parquet` also writes `results.parquet`; `results.csv` is always written |
| `output_path` | str | `{name}_{datetime}` | relative output-directory template; see [output destinations](#output-destinations); ignored for committed-manifest URI runs |

NOTE: `search_strategy.seed` seeds random sampling, Sobol sampling or a grid with `shuffle: true`. An unshuffled grid keeps its declared enumeration order. The traversal seed does not seed models. Sobol always uses scrambling and rejects additional strategy fields; see [Sobol's mapping and resume contract](Advanced-Search.md#seeded-sobol-sampling).

`pruning_strategies` declares reducers that fire every `feedback_interval` rounds during a run — see [Reducers And Feedback](Reducers-And-Feedback.md) for the reducer catalog, YAML syntax, and tuning.

### Output destinations

For direct YAML runs, `uel.output_path` is relative to `results/dev` in development
and `results` in production, beneath the invocation's current working directory.
`{name}` and `{datetime}` are substituted without rewriting the saved template.
The expanded path must be relative and resolve to a strict descendant of that
mode directory. Absolute paths, escapes through parent components or existing
symlinks, and paths naming the mode directory itself are rejected before
compilation or project writes, including with `--dry-run`.

Nested paths, contained parent components and aliases within the mode directory
remain valid. Both the mode root and destination are resolved, so a configured
symlinked results root is supported; execution uses the checked canonical path.
This checks the existing filesystem layout, not concurrent filesystem mutation.
Static `limen validate` cannot establish this filesystem-context containment.

Relocate direct-run output through the invocation's working directory or a
symlinked results root. Python callers may pass an explicit `results_base` to
`run_experiment`; its `results[/dev]` directory becomes the authorized mode root.
The YAML file's location does not change cwd-relative output or source paths.

Committed-manifest URI runs ignore the template and keep the project-root
`results[/dev]/<id8>/<timestamp>` layout, including from a project subdirectory.
Invoking the stored YAML by file path uses direct-YAML admission instead.
Explicit `--resume` uses the selected saved directory, including relocated runs
whose legacy YAML contains an absolute template. The copied YAML, metadata and
manifest identity retain the declared template. Occupied-run refusal still
precedes YAML copying.

`uel.intra_callback` is unsupported in YAML and rejected whenever present,
including null or a resolvable callable name. Use the Python
`UniversalExperimentLoop(intra_callback=callback)` argument with a callable
receiving `(log, msq)`; see [manual intervention](Developer/Pruning-Strategies.md#manual-intervention).
Fresh CLI commands and `limen run --resume` reject saved declarations containing
the key after schema-version admission and before construction or scientific
writes. Native callable callbacks and scientific read-back remain supported;
no YAML import or callback registry exists.

### Parallel rounds

Set `uel.workers: 4` or run `limen run experiment.yaml --workers 4` to evaluate
up to four rounds inside one experiment. Omitted workers and `workers: 1` use
the existing serial path. The CLI option overrides YAML; it changes execution
resources without rewriting the source YAML, `yaml_reference` or manifest hash.
Different YAML documents retain their different content-based manifest IDs.

Parallel admission requires an MSQ search, a YAML-compiled manifest, per-round
preparation and `post_processing=False`. Every built-in architecture is admitted
when the CPU knobs it exposes resolve to one thread. Where the architecture
exposes `n_jobs`, set the search parameter `n_jobs: [1]`: every effective value,
including defaults and context overrides, must be exactly 1, so the LightGBM
and XGBoost null defaults are rejected rather than silently overriding a
searched parameter. Where it exposes `device`, every value must be `cpu` or the
backend's null CPU default. Architectures without these knobs, such as DLinear,
Ridge and rule-based strategies, need no setting. Each worker verifies after
every round that all native pools ran one thread. TabPFN is neither refused nor
proven on this path. Workers use one native thread each for BLAS/OpenMP and
Polars; the parent environment is restored.
The run records `workers` and `worker_native_threads` separately in metadata.
Serial runs retain their existing metadata schema and native thread behavior.

Existing calibration, validation-return objectives, recording and walk-forward
remain usable on this path. Walk-forward folds run sequentially within each
worker's round. Parallel mode rejects custom/native SFD callables, overridden
prep/model functions, non-CPU backends, non-MSQ execution, retained live
post-processing and `factorize=True` before output creation. These combinations
remain available with one worker. There is no implicit serial fallback.

The parent freezes the learning, execution and funding observations once.
Parameter-dependent secondary sources and pipelines that cannot be reconstructed
and frozen are rejected. Spawned workers rebuild their isolated mutable manifest
from the unchanged YAML reference and consume those fixed observations. The
native entry point is `UniversalExperimentLoop.run(workers=N)` with that compiled
manifest; call it inside the normal `if __name__ == '__main__':` process guard.

Only the coordinator draws search occurrences, handles feedback and writes
artifacts. `_round_index`, together with recorded parameter identity, identifies
an occurrence even when an intervention injects a repeated parameter set. The
entire dispatched-but-uncommitted window is bounded by workers, including
completed results waiting for an earlier occurrence. CSV, JSONL, audit,
walk-forward tracks and final Parquet retain logical occurrence order.

The coordinator drains each feedback and periodic checkpoint boundary before
drawing later rounds. The first feedback trigger remains zero-based index
`feedback_interval`, after `feedback_interval + 1` terminal attempts; strict
failures count as attempts. Callbacks and intervention handling keep their existing
interfaces in the parent. Metric-driven feedback can match serial execution for
identical inputs and diagnostics. Walltime/throughput controls and externally
timed interventions retain their behavior; their decisions can differ when
elapsed times or file arrival times differ.

On the first SIGTERM/SIGINT, the coordinator stops admission, finishes any active
artifact write, saves its highest contiguous terminal prefix and matching search
state, then terminates and reaps uncommitted workers. Resume re-evaluates that
suffix from the same occurrence indices and parameters. Parallel runs save an
empty-prefix checkpoint before dispatch and a final checkpoint on graceful stop
or completion. A subsequent signal may force termination. SIGKILL or power loss
inside an artifact write has no additional recovery guarantee.

With applicable [recording evidence](Universal-Experiment-Loop.md#recording-policy) enabled on the initial run,
`limen run --resume RESULTS_DIR` reuses saved workers and thread policy; old
metadata without these fields resumes serially. An explicit changed policy is
rejected before truncation or rewriting. Committed CSV occurrences must join to
their successful JSONL/fold/recording evidence before resume rewrites anything.
Strict failures retain one CSV row and no successful prediction/fold evidence.
The checkpoint retains their typed effective parameters; resume reconstructs
successful parameters from JSONL, preserving empty strings, nulls and overrides.
Parallel metadata declares `context_parameter_keys`; resume requires those keys
in every committed parameter record before restoring typed values.
Continuing an unfinished native run requires its original public `context_params`
keys; omission or a changed declaration fails before artifacts are rewritten.
A completed run may restore its recorded values without supplying those keys.
Malformed or missing committed evidence fails loudly. Other model/worker failures
remain fatal, preserving the valid prefix and reporting identity and traceback.
Stopped/resumed equality requires unchanged source observations, deterministic
model settings and equivalent feedback inputs; concurrency does not establish
new RNG or statistical guarantees.

## Manifest Types

There are two manifest subclasses under the YAML and Python surfaces:

- **`MLManifest`** — for ML pipelines. Supports indicators, features, targets, scalers, ablation, and calibration.
- **`RuleBasedManifest`** — for rule-based pipelines. Supports indicators and features, plus `with_strategy()` for predicate-driven entry signals. Does not support scalers or ablation.

Both subclasses share a common base (`Manifest`) that owns data fetching, splitting, and model execution. The `manifest()` function in every foundational SFD returns the base `Manifest` type as a uniform interface, while internally constructing the correct subclass. `Manifest` itself is not meant to be instantiated directly — its `prepare_data()` raises `NotImplementedError`.

## Preparation fields and component APIs

The Compiled project declares scaling, feature ablation and PCA in YAML. The CLI builds their Manifest components internally:

<!-- example: manifest-native-preparation source=docs/examples/compiled-research/research.yaml region=preparation scenario=search interface=compiled role=executed -->
```yaml
    scaler:
      class: limen.scalers.RobustScaler
    feature_ablation:
      drop_count_key: feature_drop_count
      seed_key: feature_drop_seed
    pca_compression:
      enabled_param: auto_pca
      n_components_param: pca_k
```

Component authors can construct Manifest-backed SFDs through the optional Python APIs below. The Compiled user workflow is the [YAML definition and CLI execution](Compiled-Research.md); direct builder calls do not replace that interface.

Use the complete project instructions for the YAML workflow, dependencies, recorded inputs and working directory.

## How A Manifest Executes

The manifest pipeline is split-first by design. In other words, fitting never happens on the full dataset before the train, validation, and test partitions are created.

The execution order is:

1. fetch raw input data
2. optionally apply a pre-split selector
3. split into train, validation, and test
4. optionally form bars inside each split
5. apply indicators and features
6. compute fitted target parameters on train, then apply target transforms across splits
7. apply configured feature shims and ablation
8. fit the scaler on train, then apply it across splits
9. align the feature schema and optionally fit PCA on training
10. finalize a standard `data_dict` and apply its optional extension
11. run the configured model with round-specific parameters

This ordering establishes the native fitting boundary. It cannot repair a feature function that already reads future observations.

The Manifest controls its declared preparation stages. Keep feature causality, source availability and the intended label horizon explicit alongside those controls.

There is one more protection step after feature engineering: non-empty splits are aligned to a shared column set before the final `data_dict` is built. If a transform such as `fractional_diff` produces a column in one split but not another because the shorter split lacks enough history, Limen drops that extra column from the non-empty splits so the downstream model still receives a consistent schema.

## Data Source Configuration

### `set_data_source(method, params=None)`

Configure the production data source for the manifest.

<!-- reference: manifest-source source=limen/experiment/manifest_core.py region=signature:Manifest.set_data_source role=signature -->
```python
def set_data_source(self, method: Callable[..., Any], params: dict[str, Any] | None=None) -> 'Manifest':
    ...
```

### `set_test_data_source(method, params=None)`

Configure the test data source used when `test_mode=True` is passed to `UniversalExperimentLoop`.

<!-- reference: manifest-test-source source=limen/experiment/manifest_core.py region=signature:Manifest.set_test_data_source role=signature -->
```python
def set_test_data_source(self, method: Callable[..., Any], params: dict[str, Any] | None=None) -> 'Manifest':
    ...
```

### Data source selection

Both source setters accept synchronous frame-returning callables, including
partials, callable instances, local functions and static methods. Fetch invokes
the configured callable once with its keyword parameters and preserves its result
and exceptions. From a repository checkout, a partial of `polars.read_parquet`
bound to `tests/fixtures/spot_1h_20240101_20241231.parquet` can be passed to
`set_data_source()` or `set_test_data_source()`; `params={'n_rows': 3}` selects
the first three recorded rows through that callable's own keyword argument.

Native bound methods retain owner `.data` precedence. Genuine unbound instance
methods construct an owner only when their receiver is absent from configured
keywords; the method must populate `.data`. Wrapping a stateful method does not
adapt a `None` or row-list result into a frame. Under an active holdout reservation,
production fetching still requires the admitted native source identity; partials
and copied-metadata wrappers do not inherit that admission. Callable compatibility
also preserves deepcopy and parallel-execution admission requirements.

Pass `test_mode=True` to `UniversalExperimentLoop` to fetch from the test data source instead of the production source.
When `test_mode=True` and a test data source is configured, `fetch_test_data()` is called; otherwise `fetch_data()` is used.

That is why foundational SFDs can run locally with no explicit `data=` and still use the configured test data source.
To keep local runs bounded, configure `set_test_data_source()` with explicit `kline_size` and `row_count_limit`.

Direct protected `run_model()` calls require the same Manifest instance and its completed preparation, current reservation revision, pipeline and consumed inputs. The research lease remains held through model evaluation. Altered or foreign preparations fail before the architecture runs. Feature extensions must preserve ordered row and label carriers and declare the actual consumed feature schema. Search hooks retain the same admitted source and revision; protection failures propagate through feedback handling. The [guarded holdout reference](Guarded-Holdout-Reference.md) describes the connected research modes.

## Pipeline Configuration

### `set_split_config(train, val, test)`

Configure the relative split sizes.

<!-- reference: manifest-split-ratio source=limen/experiment/manifest_core.py region=signature:Manifest.set_split_config role=signature -->
```python
def set_split_config(self, train: int, val: int, test: int) -> 'Manifest':
    ...
```

This means 8/11 train, 1/11 validation, and 2/11 test.

Behavior rules:

- all three values must be non-Boolean Python `int` instances (integer subclasses remain supported)
- `train` must be positive
- `val` and `test` can be zero but not negative
- the method raises `ValueError` if those constraints are violated

Allowing zeros supports explicit workflows where validation or test data is intentionally absent. For example, a caller-created clone with `split_config=(1, 0, 0)` fits on all available rows. `Trainer` does not apply that override: it replays the original split so it can compare the original metrics.

`set_split_config` and `with_params_override(split_config=(...))` share this
element/range admission and reject before changing the source manifest. Integral
floats, NumPy integer scalars, NaN and infinity are refused as well as fractions
and Booleans. The override still requires a three-tuple and clears `split_dates`
on its returned copy; the direct setter retains the current date declaration.

### `set_split_dates(train_start, train_end, val_start, val_end, test_start, test_end, *, val_predict_guard=True, test_predict_guard=True)`

Pin the train, val, and test windows to absolute `datetime` bounds, in preference to ratio-based splits whose absolute boundary depends on the input row count.

<!-- reference: manifest-split-dates source=limen/experiment/manifest_core.py region=signature:Manifest.set_split_dates role=signature -->
```python
def set_split_dates(self, train_start: date | Any, train_end: date | Any, val_start: date | Any, val_end: date | Any, test_start: date | Any, test_end: date | Any, *, val_predict_guard: bool | Any=True, test_predict_guard: bool | Any=True) -> 'Manifest':
    ...
```

Behavior rules:

- Windows are half-open `[start, end)`. A row enters a split iff its `datetime` falls inside that window.
- Ordering must satisfy `train_start <= train_end <= val_start <= val_end <= test_start <= test_end`. Gaps between adjacent windows are allowed; rows inside a gap are intentionally excluded from all three splits.
- Every bound must be a `date` or `datetime` instance. Strings, ints, and floats raise `TypeError` at the API boundary.
- `val_predict_guard` and `test_predict_guard` are keyword-only and default `True`. They control whether the served `Sensor` masks predictions inside the val and test windows. Both `True` masks the full `[train_start, test_end)` envelope — every served prediction is identical to omitting them; setting one to `False` makes that window emit real `0/1` predictions instead of `None`, so a served cohort can be checked against its own test-split backtest. Train is always masked (the model trains on it) and has no flag. Each flag must be a `bool` or `set_split_dates` raises `TypeError`.
- When `split_dates` is set it takes precedence over `set_split_config` in both `prepare_data` and `compute_test_bars`. The split runs on the raw data (before per-split feature transforms), so transforms can still drop rows inside a slice but cannot move rows across slice boundaries.
- `with_params_override(split_config=(1, 0, 0))` clears any previously pinned `split_dates`, so an explicit ratio override is not silently shadowed by an earlier date pin.

Preparation requires admitted training rows and at least one row in each positive-duration validation or test window. Equal-bound future windows remain intentional zeros. After holdout admission and pre-split selection, any required raw window without rows raises a named `ValueError` before target/scaler fitting. Bar formation or later processing that exhausts a required window raises `StrictModeError`; UEL records a failed candidate, including when `strict_mode` is false. Retained rows are checked after context removal and before PCA/model consumption. `compute_test_bars` checks the required test window without requiring the other windows; standalone `split_by_dates` keeps its slicing contract.

Use this when the train / val / test boundaries must land on specific datetimes (e.g. honoring a deployment date), and `set_split_config` when proportions of the row count are the natural way to express the split.

Direct `HistoricalData.get_spot_klines` declarations add a native-bar admission rule: `train_end`, `val_end` and `test_end` must each lie on the UTC Unix-epoch grid of the effective `kline_size` (60 seconds when absent or `None`). Preparation and `compute_test_bars` check both learning and test-source declarations, including an unused declaration. Bound methods and unchanged inherited methods count. Dates mean midnight UTC; datetimes must be timezone-aware and are checked by their UTC instant. An off-grid end raises `ValueError` even when a truncated fetch or gap contains no crossing bar; replay of an incompatible saved declaration also refuses.

For YAML date bounds, use an interval that divides a day or choose epoch-grid-aligned window ends. Starts and excluded gap rows retain their row-label behavior, and standalone fetching can still return partial edge buckets. This check trusts the declared native interval; it does not certify supplied frames or constituent completeness. File sources, dollar bars, opaque wrappers and custom preparation retain their existing row-label semantics.

In a YAML manifest the same windows and flags live in the `split_dates` block. `val_predict_guard` and `test_predict_guard` are optional and default to `true`; a `train_predict_guard` key (or a non-boolean value for either flag) fails `limen validate`:

| Guard field | Behavior |
|---|---|
| `val_predict_guard` | True keeps pre-test validation inference guarded; explicitly false allows the validation window. |
| `test_predict_guard` | True keeps pre-test inference guarded; explicitly false allows earlier inference under the caller's declared protocol. |

### Walk-forward sweeps

Use `sfd.manifest.split_walk_forward` instead of `split_dates` to evaluate every parameter trial across successive test windows:

<!-- example: manifest-walk-definition source=docs/examples/compiled-research/walk_forward.yaml region=__file__ scenario=walk_forward interface=compiled role=executed -->
```yaml
# region declaration
schema_version: '1.0'
metadata:
  name: book_compiled_walk_forward
  mode: development
sfd:
  manifest:
    type: ml
    data_source:
      method: limen.data.HistoricalData.get_any_file
      params:
        file_path_or_url: inputs/spot_1h_20240101_20240116.parquet
    indicators:
    - func: limen.indicators.roc
      params:
        period: 1
        group: momentum
    target:
      name: quantile_flag
      class: limen.targets.QuantileBinaryTarget
      fit_params:
        source_column: roc_1
        quantile: 0.5
      transform_params:
        shift: -1
    backtest:
      fee_bps: 7.0
      slip_bps: 3.0
      notional_rate: 0.5
      product:
        kind: cash_spot
        instrument: BTCUSDT
        base_currency: BTC
        quote_currency: USDT
        quantity_step: 1e-09
        min_notional: 0
      source_interval_seconds: 3600
    scaler:
      from_params: scaler_type
    reference_architecture: limen.sfd.reference_architecture.logreg_binary
    split_walk_forward:
      n_folds: 2
      test_bars: 48
      purge_bars: 4
      embargo_bars: 3
      anchored: true
    calibration:
      probability_calibration:
        func: limen.calibration.sklearn_probability_calibrator
        params:
          method: sigmoid
      threshold_function:
        func: limen.calibration.grid_threshold_optimizer
        params: {}
    objective:
      metric: backtest_total_return
      direction: maximize
  params:
    scaler_type: [logreg]
    C:
    - 1.0
    random_state: [42]
uel:
  record_execution: true
  output_path: compiled-walk_forward
  n_permutations: 1
  search_strategy:
    type: grid
  output_format: csv
  checkpoint_interval: 1
  recording:
    predictions: true
    alignment: true
    trade_ledger: true
    fold_details: true

assessment:
  metric: val_backtest_total_return
  direction: maximize
  top_k: 1
  folds: [0, 1]
  predict: true
  inference_source: inputs/spot_1h_20240101_20241231.parquet
# endregion declaration
```

The native equivalent is:

<!-- reference: manifest-walk-method source=limen/experiment/manifest_core.py region=signature:Manifest.set_split_walk_forward role=signature -->
```python
def set_split_walk_forward(self: _TManifest, *, n_folds: int, test_bars: int, purge_bars: int, embargo_bars: int, anchored: bool) -> _TManifest:
    ...
```

All five fields are required. `n_folds` must be at least `2`, `test_bars` at least `1`, and `purge_bars` and `embargo_bars` nonnegative integers; booleans are not integers here. `anchored` must be a boolean. The declaration cannot coexist with `split_dates` and cannot contain parameter references or extra fields. Data-source date limits belong in `data_source.params`; they are not derived from fold positions.

The final `n_folds * test_bars` source rows form consecutive test windows. Each fold's preceding training pool excludes the `purge_bars` rows immediately before its test window and the `embargo_bars` rows after earlier test windows. Anchored pools start at the beginning of the source; rolling pools keep the first fold's positional span before embargo removal.

The existing train-to-validation ratio divides each pre-test pool; the default is `8:1`. A further `purge_bars` gap separates model training from validation. Choose a purge at least as long as the target's forward label horizon. Insufficient rows fail before fitting.

Validation-return objectives accept compatible ML and rule-based manifests, including equivalent architecture wrappers. The objective scores aligned validation decisions under the declared event execution policy. A fitted predictor is called with fresh validation-only inputs, or an architecture supplies `_validation` with the ordered `row_ids`, `source_binding` and `prediction_mode` received in `data['_validation_context']`, plus its actual `predictions`. Explicit evidence is authoritative; malformed or stale evidence fails. Probabilities are required when active calibration or threshold selection consumes them. The effective calibration configuration is bound whether supplied by the manifest or the architecture call. Active calibration must fit the original ordered validation operands; threshold callbacks retain those original labels even when probability calibration is disabled. Scored decisions must use the resulting calibrated probabilities and selected threshold. Allocation scores its bound outer validation exposure batch.

Both ML and rule-based manifests support walk-forward sweeps. Each fold prepares its own partitions and fits its configured target, scaler and model afresh. Probability calibration, threshold selection and the validation-return objective use only that fold's validation partition; their existing configuration and flags still control whether tuning runs. Test execution uses the fold's fitted model and retained test-row order. No fitted state carries from one fold to another.

One trial consumes one permutation and reaches feedback only after its folds finish. Its numeric result values are the arithmetic means of finite fold values; these are the values existing reducers see. The individual fold results and test net-return track retain the evidence behind that aggregate. A mean of fold total returns is not a compounded return for the stitched test period.

`uel.fold_results` keeps the individual result rows with trial `id` and `fold` identity. The aggregate `optimal_threshold` is null: each fold has its own fitted threshold, retained in its result. Post-run `Log` backtest rows remain per fold rather than treating the concatenated test windows as one continuous position path. Prediction and confusion replay that assumes one test set rejects walk-forward trials.

In saved CLI or Python/MSQ runs, `round_data.jsonl.gz` omits nested `folds` by default. Enable `uel.recording.fold_details` to retain each fold's results and net returns; enable `predictions` and `alignment` for its row arrays. The [recording policy](Universal-Experiment-Loop.md#recording-policy) can omit the entire fold payload or selected fields within retained folds without changing `trial_returns.parquet` or DSR. PBO requires retained fold/timestamp evidence; its selection constraints are described below. `trial_returns.parquet` contains exactly `trial`, `bar` and `net_return`, with one row group per successful trial. Legacy standard runs write it in `experiment_dir`, or beside the named experiment output when no directory is supplied. `bar` is a consecutive ordinal over the actual retained test rows, in fold order; it adds no rows for excluded or dropped bars.

Every fold starts execution independently. Snapshot net returns are the existing notional-scaled `execution.net` values. Event execution samples its recorded equity at retained test-bar endpoints, including the terminal mark in the final value; each return is current equity divided by prior equity minus one, starting from the policy's initial equity. The return track is captured for walk-forward trials even when `record_execution` is off; that flag retains its existing optional JSONL fields. Existing snapshot metrics and their columns remain unchanged.

Walk-forward resume requires predictions, alignment and fold details; event execution additionally requires its ledger. It rejects a changed walk-forward declaration or native train-to-validation ratio and missing applicable committed payloads before rewriting artifacts. A matching resume restores the saved nested fold records and regenerates the Parquet return track without replaying execution. Promotion separately requires [versioned fold evidence](Trainer.md#recorded-walk-forward-folds). New single-split runs use the same lean defaults; existing full logs retain resume behavior.

These folds provide recorded out-of-sample evaluations. The selection report below assesses the recorded sweep; its statistics provide no guarantee against selection optimism or future loss. Omitting `split_walk_forward` preserves the existing single-split path.

### Acceptance thresholds

Walk-forward runs with `trial_returns.parquet` produce `acceptance_report.json` and `acceptance_report.md` beside that artifact. To add threshold verdicts, declare `sfd.manifest.acceptance`:

| Acceptance field | Meaning |
|---|---|
| `min_deflated_sharpe_probability` | Finite minimum probability in `[0, 1]`. |
| `max_pbo` | Finite maximum probability in `[0, 1]`. |
| `pbo_blocks` | Representable even integer of at least two, dividing the retained bar count exactly and leaving at least two bars per block. |

The native equivalent, after `set_split_walk_forward(...)`, is:

<!-- reference: manifest-acceptance source=limen/experiment/manifest_core.py region=signature:Manifest.set_acceptance role=signature -->
```python
def set_acceptance(self: _TManifest, *, min_deflated_sharpe_probability: float | None=None, max_pbo: float | None=None, pbo_blocks: int | None=None) -> _TManifest:
    ...
```

Each key is optional, but the block must contain a probability threshold or `pbo_blocks`. Thresholds must be finite literal numbers in `[0, 1]`. The optional partition count must be a literal even built-in integer of at least two; booleans, floats, strings, parameter references, odd counts and smaller counts reject. The count must fit the current interpreter's decimal representation used by existing JSON persistence; an unrepresentable count raises before artifacts or fitting. Omission defaults to two; there is no automatic choice or fallback. An acceptance declaration requires `split_walk_forward`.

For example, `acceptance: {pbo_blocks: 8}` selects eight without adding a threshold. The native equivalent is `.set_acceptance(pbo_blocks=8)`. An explicit count (including two) requires both `search_strategy` and `experiment_dir`, with `recording.alignment` and `recording.fold_details` enabled, before run artifacts. Count-only reporting needs only those two payloads; resumable runs additionally require predictions and any applicable event ledger. Omitted-count and existing nonpersisted threshold-only runs retain their behavior.

Selecting `max_pbo` or an explicit `pbo_blocks` also requires `recording.fold_details` and `recording.alignment` to be explicitly `true`; a conflicting policy raises before artifacts are created. With neither setting explicit, either group may be omitted: return tracks and DSR remain available, while PBO is `null` with an explanation. Direct `acceptance_report(..., acceptance={"max_pbo": ...})` or an explicit count rejects an incompatible saved policy before writing reports. Explicit full recording retains existing behavior; see the [recording policy](Universal-Experiment-Loop.md#recording-policy).

The report selects the highest per-bar Sharpe among successfully recorded trial tracks; exact ties prefer the first persisted trial. DSR uses that successful trial count and the sample variance (`ddof=1`) of their Sharpes. PBO uses the declared count of contiguous equal blocks over the actual retained track, with no discarded tail. There must be at least two trials, exact divisibility and at least two bars per block. Blocks need not coincide with folds; there is no extra block-boundary purge/embargo. Every half still contains `n_bars / 2` observations. The default two evaluates only two balanced combinations. Eight uses 70 combinations and a nominal counting step of `1/70`; ten and twelve require 252 and 924 combinations. With 120 retained bars, both ten and twelve satisfy the geometry. More combinations create no new observations or independence. Explicit large counts can make exhaustive reporting computationally impractical; see the [cost and resolution discussion](Benchmark.md#walk-forward-selection-report) before choosing a partition.

The JSON fields are `n_trials`, `n_bars`, `n_blocks`, `pbo_combinations`, `pbo_block_bars`, `pbo_resolution`, `winner_trial`, `winner_sharpe`, `trial_sharpe_variance`, `deflated_sharpe_probability`, `pbo`, `thresholds`, `verdicts` and `errors`. A DSR verdict passes at or above its minimum; a PBO verdict passes at or below its maximum. Without an acceptance block, `verdicts` is empty and the statistics are still reported. A failed verdict is `false` and never aborts the run. An unavailable statistic is `null` with an explanation in `errors`; its declared verdict is also `null`, rather than a pass. See [Benchmark](Benchmark.md#walk-forward-selection-report) for the metric definitions, direct-function input errors and interpretation limits.

New walk-forward metadata saves effective `pbo_blocks`; the report calls that value `n_blocks`. Live manifest, compiled YAML and a supplied `yaml_reference` must agree on the count before writes (omission means two). Compiled-only runs retain their YAML. Resume rejects a changed count; present saved metadata and YAML counts must be valid and agree. Missing metadata or a missing count means legacy two. Direct regeneration inherits saved S when acceptance is omitted or contains thresholds only; an explicit different count raises before changing any artifact. The partition is frozen; thresholds remain caller-supplied on regeneration, as today. Omitting acceptance drops verdicts. Invalid saved counts raise rather than becoming unavailable PBO. Older checkouts that enforce the former eight-block cap reject newly admitted saved counts; no existing count is rewritten.

`pbo_combinations` and `pbo_resolution` describe nominal enumeration, even when evidence makes PBO unavailable. Unknown or invalid retained geometry makes all three geometry fields null before combinations are computed. For valid geometry, an exact combination count beyond the interpreter's decimal representation makes combinations and resolution null with `errors.pbo_geometry`; bars/block remains available. Otherwise the exact integer denominator is retained and `pbo_resolution` is its floating reciprocal: `0.0` means floating underflow, not an exact zero step. Sparse trials can have supported DSR but zero variance in a finer half sample; PBO and its verdict then remain null, with no count fallback. Null is not pass.

The canonical Compiled walk-forward case uses its complete project definition
shown above. It restricts the source to 384 recorded hourly bars, executes the
configured folds, reconstructs each selected fold and binds a qualified Cohort:

<!-- example: manifest-walk-execution source=docs/examples/compiled-research/run.sh region=walk_forward scenario=walk_forward interface=compiled role=executed -->
```bash
limen run --no-progress-bar compiled-research/walk_forward.yaml
limen assess compiled-research/walk_forward.yaml --results results/dev/compiled-walk_forward
limen run --no-progress-bar compiled-research/acceptance.yaml
```

### Recorded acceptance example

The Compiled project's walk-forward scenario also executes this rule-based
comparison on the complete recorded hourly source. Two entry thresholds produce
two trials with two folds each. The retained tracks contain 96 returns per trial;
the default CSCV partition uses two blocks of 48 returns.

The declaration uses its literal bundled input path. Run it from the bundle root.

<!-- example: manifest-recorded-acceptance source=docs/examples/compiled-research/acceptance.yaml region=__file__ scenario=walk_forward interface=compiled role=executed -->
```yaml
# region declaration
schema_version: "1.0"
metadata:
  name: recorded_acceptance
  mode: development
sfd:
  manifest:
    type: rule_based
    data_source:
      method: limen.data.HistoricalData.get_any_file
      params:
        file_path_or_url: inputs/spot_1h_20240101_20241231.parquet
    split_walk_forward:
      n_folds: 2
      test_bars: 49
      purge_bars: 4
      embargo_bars: 3
      anchored: true
    acceptance:
      min_deflated_sharpe_probability: 0.95
      max_pbo: 0.20
    indicators:
      - func: limen.indicators.window_return
        params:
          period: 1
    strategy:
      conditions:
        - id: entry
          name: positive_return
          type: threshold
          column: ret_1
          operator: ">"
          value: "{entry_return}"
      entry: entry
    reference_architecture: limen.sfd.reference_architecture.rule_based
    backtest:
      fee_bps: 7.0
      slip_bps: 3.0
      notional_rate: 0.5
  params:
    entry_return: [0.0, 0.001]
uel:
  record_execution: true
  output_path: compiled-acceptance
  recording:
    alignment: true
    fold_details: true
  n_permutations: 2
  search_strategy:
    type: grid
  prep_each_round: true
  output_format: parquet
# endregion declaration
```

The CLI retains `trial_returns.parquet`, `acceptance_report.json`,
`acceptance_report.md` and the fold records under `results/dev/compiled-acceptance/`.
The independent project proof checks the timestamp population, recomputes the
selected trial's Deflated Sharpe probability and both CSCV rankings, and compares
the JSON and Markdown verdicts. Failed verdicts leave the research results
available. Null statistics remain unavailable rather than passing.
For report definitions and family comparisons, continue to [Benchmark](Benchmark.md#walk-forward-selection-report).

### `set_pre_split_data_selector(func, **params)`

Optionally select or reduce the raw dataset before splitting.

<!-- reference: manifest-pre-selector source=limen/experiment/manifest_core.py region=signature:Manifest.set_pre_split_data_selector role=signature -->
```python
def set_pre_split_data_selector(self, func: Callable[..., pl.DataFrame], **params: Any) -> 'Manifest':
    ...
```

Use this for smaller or controlled slices of the raw dataset before the normal split-first pipeline begins.

`random_slice` requires `rows` to resolve to a positive, non-boolean integer; invalid row counts raise `ValueError` before selecting data.

### `set_bar_formation(func, **params)`

Configure threshold-bar formation inside each split.

<!-- reference: manifest-bars source=limen/experiment/manifest_core.py region=signature:Manifest.set_bar_formation role=signature -->
```python
def set_bar_formation(self, func: Callable[..., pl.DataFrame], **params: Any) -> 'Manifest':
    ...
```

See [Data Bars](Data-Bars.md) for the supported bar types and output schema.

### `set_required_bar_columns(columns)`

Declare the downstream columns that bar preparation must retain. After any configured bar formation completes, or directly for base bars, preparation first reads `datetime`, whether or not it was declared; its absence raises Polars `ColumnNotFoundError`. If that read succeeds, any missing declared column raises `AssertionError` under both normal Python and `python -O`.

An explicit setter or YAML `required_columns` declaration must be a nonempty list
of distinct nonblank strings. Non-list values and non-string entries raise
`TypeError` at the native setter; empty, blank or duplicate entries raise
`ValueError`. YAML validation locates the field or offending item, and compilation
uses the same admission. An explicit `null` is rejected; omit the declaration to
keep the default without additional membership requirements. Names retain exact
case, padding and order; blank means empty or whitespace-only. The setter stores
an owned list copy, so later caller-list edits do not change the declaration.
Constructor field overrides and direct attribute assignment retain their existing
behavior outside these setter/YAML declaration checks.

<!-- reference: manifest-required-bars source=limen/experiment/manifest_core.py region=signature:Manifest.set_required_bar_columns role=signature -->
```python
def set_required_bar_columns(self, columns: list[str]) -> 'Manifest':
    ...
```

This assertion protects model or backtest paths that require OHLC fields after bar formation.

## Indicators And Features

`add_indicator()` and `add_feature()` use the same underlying mechanism. Both add transformation steps to the split-first pipeline.

### `add_indicator(func, group=None, include_if=None, **params)`

<!-- reference: manifest-indicator source=limen/experiment/manifest_core.py region=signature:Manifest.add_indicator role=signature -->
```python
def add_indicator(self, func: Callable[..., Any], group: str | None=None, include_if: str | None=None, **params: Any) -> 'Manifest':
    ...
```

### `add_feature(func, group=None, include_if=None, **params)`

<!-- reference: manifest-feature source=limen/experiment/manifest_core.py region=signature:Manifest.add_feature role=signature -->
```python
def add_feature(self, func: Callable[..., Any], group: str | None=None, include_if: str | None=None, **params: Any) -> 'Manifest':
    ...
```

### Parameter resolution

Manifest parameters are resolved at run time:

- literal scalars are passed through unchanged
- a string matching a `round_params` key is looked up from the current round
- a formatted string like `'roc_{roc_period}'` is interpolated from `round_params`
- strings starting with `_` are treated as fitted-parameter references when available

A native parameter value such as `period="roc_period"` resolves through the current round's `roc_period` key.

The resolved value is passed to the indicator's `period` argument before that transform runs.

### Group filtering

Use `group=` to tag transforms into families, then filter by `feature_groups` in `round_params`.

| Group declaration | Selection |
|---|---|
| `group="momentum"` | Included by `feature_groups="momentum"` or `all`. |
| ungrouped | Always included. |

If `round_params['feature_groups']` is present, only transforms whose group is in that string are applied. Ungrouped transforms always run. The value is a pipe-delimited string of active group names, or `'all'` to include every group.

### Conditional inclusion

Use `include_if=` when a transform should only run if a boolean round parameter is true. Missing control keys are treated as false.

Set `include_if` to the boolean round-parameter key that controls the feature.

## Parameter-Controlled Perturbations

The manifest builder supports perturbation-style workflows directly in the declarative surface.

### Feature-group selection

Use `group=` on indicators or features, then pass `feature_groups` in `round_params`.

Tag indicators/features through the source-bound methods above; select active tags through the pipe-delimited `feature_groups` round value.

At run time:

`feature_groups="momentum"` selects momentum-tagged and ungrouped transforms.

Use `|` to activate multiple groups: `'momentum|microstructure'`. Use `'all'` to include every group.

Only grouped transforms in the selected families run. Ungrouped transforms always run.

### Conditional feature toggles

Use `include_if=` for boolean on/off switches:

The feature's `include_if` names its boolean controlling parameter.

At run time:

`use_vwap=False` disables a feature declared with `include_if="use_vwap"`.

The transform is skipped.

### Feature ablation

Use `set_feature_ablation()` to let the manifest randomly drop feature columns after transforms and before scaling.

<!-- reference: manifest-ablation-introduction source=limen/experiment/manifest_core.py region=signature:MLManifest.set_feature_ablation role=signature -->
```python
def set_feature_ablation(self, drop_count_key: str='feature_drop_count', seed_key: str='feature_drop_seed') -> 'MLManifest':
    ...
```

Then control it from round parameters:

| Round key | Project value |
|---|---|
| `feature_drop_count` | Baseline 0, compressed case 1. |
| `feature_drop_seed` | 7. |

Important behavior:

- the manifest mutates `round_params` by adding `_dropped_features`
- protected columns such as `datetime` and the target are not eligible
- the same seed reproduces the same dropped columns

The selected column depends on the eligible feature set. The exact list is recorded in `round_params['_dropped_features']`.

In saved CLI or Python/MSQ runs, that `_dropped_features` payload is stored into `round_data.jsonl.gz`. [Trainer](Trainer.md) reproduces the same drop set when the saved YAML definition supports reconstruction.

### Scaler choice from parameters

Use `set_scaler_from_params()` when scaler selection itself is part of the search space.

<!-- reference: manifest-dynamic-scaler-introduction source=limen/experiment/manifest_core.py region=signature:MLManifest.set_scaler_from_params role=signature -->
```python
def set_scaler_from_params(self, param_name: str='scaler_type', extra_params: dict[str, Any] | None=None) -> 'MLManifest':
    ...
```

The registry resolves:

- `RobustScaler` when `scaler_type='robust'`
- `RankGaussScaler` when `scaler_type='rank_gauss'`

### Manifest-level overrides

Use `with_params_override(split_config=(1, 0, 0))` for a manifest clone with structural changes that should not come from the round search itself.

Examples:

- `split_config=(1, 0, 0)` for an explicit caller-owned all-data fit
- `start_date_limit='2025-01-01'` for a controlled data-window variant
- `row_count_limit=5000` for test or smoke paths

Round parameters handle search-time variation. `with_params_override(split_config=(1, 0, 0))` handles external structural control.

Declare boolean candidates for the controlled feature; use a derived Manifest for build-time source/split changes.

## Target Configuration

Target construction uses `Manifest.with_target_label()` with a class from `limen.targets`. The class is fitted once on the training split and then applied to validation and test without refitting.

<!-- reference: manifest-target source=limen/experiment/manifest_core.py region=signature:Manifest.with_target_label role=signature -->
```python
def with_target_label(self, target_name: str, target_class: type, fit_params: dict[str, Any] | None=None, transform_params: dict[str, Any] | None=None) -> 'Manifest':
    ...
```

`fit_params` are forwarded to `__init__` on the training split. `transform_params` are forwarded to `transform()` on every split. Both support round-param references (`'q'` resolves to the current round's `q` value).

The target column is placed last before Limen finalizes the `data_dict`.

See [Targets](Targets.md) for the full reference including all built-in target classes and the custom target class convention.

## Scaling

### `set_scaler(transform_class, param_name='_scaler', extra_params=None)`

Configure a fitted scaler that is instantiated on train and then applied across the splits.

<!-- reference: manifest-scaler source=limen/experiment/manifest_core.py region=signature:MLManifest.set_scaler role=signature -->
```python
def set_scaler(self, transform_class: Any, param_name: str='_scaler', extra_params: dict[str, Any] | None=None) -> 'MLManifest':
    ...
```

The fitted scaler is stored in the resulting `data_dict` under `_scaler`.

Pass `extra_params` to forward constructor arguments. String values are treated as sweep param references resolved from `round_params` at fit time; all other values are static:

Underscore-prefixed strings, such as `_scaler`, resolve to fitted parameters only when that key is available in the current fitted-parameter map. Otherwise they remain literal strings.

<!-- reference: manifest-causal-scaler source=limen/scalers/causal_rolling_robust_scaler.py region=signature:CausalRollingRobustScaler.__init__ role=signature -->
```python
def __init__(self, x_train: pl.DataFrame, window: int=1000, quantile_range: tuple[float, float]=(0.25, 0.75), clip: float=8.0, min_samples: int=50) -> None:
    ...
```

### `set_scaler_from_params(param_name='scaler_type', extra_params=None)`

Select a scaler dynamically from `SCALER_REGISTRY` using a round parameter.

<!-- reference: manifest-scaler-choice source=limen/experiment/manifest_core.py region=signature:MLManifest.set_scaler_from_params role=signature -->
```python
def set_scaler_from_params(self, param_name: str='scaler_type', extra_params: dict[str, Any] | None=None) -> 'MLManifest':
    ...
```

Declare scaler registry names as candidates for the selected round key; use [Scalers](Scalers.md) for all supported names and fitting behavior.

Use this when scaler choice is itself part of the search space. Pass `extra_params` to forward constructor arguments using the same static/dynamic resolution as `set_scaler()`.

### `set_strict_mode(strict_mode)`

Enable strict null checking after the Context Carry-Over (CCO) pipeline.

<!-- reference: manifest-strict source=limen/experiment/manifest_core.py region=signature:MLManifest.set_strict_mode role=signature -->
```python
def set_strict_mode(self, strict_mode: bool) -> 'MLManifest':
    ...
```

#### Background: Context Carry-Over

When `prepare_data` processes val and test splits, it prepends a raw tail of the preceding split (the CCO block) before running feature transforms and the scaler. This ensures indicator warm-up rows and rolling-scaler warm-up rows are fully warmed before the first real split row is produced — matching how the Sensor sees a continuous stream in live inference.

After the CCO rows are stripped from each processed split, two null checkpoints run on feature columns only (the target column, which always has a trailing null from `shift:-1`, is excluded):

- **Checkpoint A** — after leading warm-up nulls are sliced off, before the scaler. Nulls here indicate mid-split data gaps in the raw input or indicator failures.
- **Checkpoint B** — after the scaler and after CCO rows are removed. Nulls here would indicate the scaler itself introduced them (rare).

Checkpoint A also runs on the training split, where there is no CCO block.

#### Strict vs non-strict

`strict_mode=True` — any unexpected null raises `StrictModeError` with the split name, checkpoint label, column name, and the timestamps of the null rows. The UEL catches this per-permutation, records the error message in `results.csv` under the `strict_mode_error` column (metric columns are empty for that round), and continues to the next round.

`strict_mode=False` (default) — the same detection runs, but logs a `WARNING` instead of raising. Training continues and the final `drop_nulls()` silently removes the affected rows.

The native constructor, setter and YAML declaration require a Boolean. Strings (including `"false"`), numeric values and explicit null raise configuration errors before preparation. YAML Boolean anchors and aliases are accepted. Missing strict mode remains false.

All foundational ML SFDs and YAML templates ship with `strict_mode=True`. New SFDs should follow the same convention — silent drops during search can mask data quality problems that only appear under specific scaler or param combinations.

## PCA Compression

### `set_pca_compression(enabled_param='auto_pca', n_components_param='pca_k', scaler_param_name='_scaler', component_prefix='pc_')`

Configure optional PCA feature compression for ML manifests.

<!-- reference: manifest-pca source=limen/experiment/manifest_core.py region=signature:MLManifest.set_pca_compression role=signature -->
```python
def set_pca_compression(self, enabled_param: str | Any='auto_pca', n_components_param: str | Any='pca_k', scaler_param_name: str | Any='_scaler', component_prefix: str | Any='pc_') -> 'MLManifest':
    ...
```

When `round_params[enabled_param]` is absent or `False`, the manifest uses the
current scaled feature surface unchanged.

When it is `True`, `round_params[n_components_param]` must be an integer. The
manifest fits full-SVD PCA on the train split only, then applies that frozen
rotation to validation and test. `x_train`, `x_val`, and `x_test` are replaced
with `pc_*` columns; no parallel raw-feature dataset is retained.

The canonical project uses `RobustScaler`. Fixed or parameter-selected
`LogRegScaler`, `LinearScaler`, `RankGaussScaler`, `CausalRollingRobustScaler`
and caller scalers also work. Configure a scaler with a fitted `transform`,
use its actual fitted-output key as `scaler_param_name`, and supply numeric,
finite finalized feature matrices in their existing order. PCA fits only on
training rows. Invalid finalized values name the split index and offending
columns before PCA construction. PCA does not clean the supplied matrix.
Existing strict null refusals, non-strict NaN-row cleanup and conditional-fold
errors remain in place.
Frozen Sensor preprocessing retains its existing null-row/error policy.

PCA weights the matrix produced by the caller's preprocessing. A scaler may
leave some columns in their original units; numeric admission does not make
those columns comparably scaled or establish strategy quality. The canonical
preparation case retains its `RobustScaler` training medians and interquartile
ranges, prepared matrices and fitted PCA projection. Its independent proof
checks those values and the unchanged training fit used on validation and test.
Rolling scalers still need their declared historical context, including at
both future split boundaries. See [Scalers](Scalers.md) for each scaler's
transformed and preserved columns.
PCA does not test stationarity. Optional [feature shims](#feature-shims) transform
selected columns upstream; their training evidence does not guarantee stationarity.

## Rule-Based Strategy

Use `with_strategy()` when the strategy is expressed as boolean predicate logic over indicator columns — no Python model code required. This path is for rule-based SFDs only and produces a different `data_dict` shape than the ML path.

### `with_strategy(conditions, entry)`

Configure a rule-based strategy from a list of condition config dicts and an entry signal id.

<!-- reference: manifest-strategy source=limen/experiment/manifest_core.py region=signature:RuleBasedManifest.with_strategy role=signature -->
```python
def with_strategy(self, conditions: list[dict[str, Any]], entry: str) -> 'RuleBasedManifest':
    ...
```

`with_strategy()` validates the config at construction time — it raises `ValueError` if any condition is missing an `id`, if ids are not unique, if `entry` does not refer to a known id, or if compound conditions reference unknown operand ids.

### Condition schema

Each condition is a dict with an `id` field; YAML records also require `name`. Put predicate arguments directly on the condition; a condition-level `params` wrapper is not consumed and is rejected in YAML. There are two kinds:

**Leaf conditions** — evaluate directly against data columns. The `type` field determines which predicate function is used:

| type | required fields | description |
|------|----------------|-------------|
| `threshold` | `column`, `operator`, `value` | column compared against a constant |
| `relative` | `column`, `operator`, `other_column` | column compared against another column |
| `crossover` | `column`, `other_column` | cross event, True on the bar of the cross; optional `direction` defaults to `above` (`above`/`below`) |
| `slope` | `column` | column rising or falling; optional `direction` defaults to `rising` (`rising`/`falling`), and `lookback` defaults to `1` |
| `sql_expr` | `expr` | SQL expression string — column names referenced directly, no `pl.col()` wrapper needed |

**Compound conditions** — combine leaf or other compound conditions using boolean logic:

| Compound operator | Fields |
|---|---|
| `and`, `or` | `id`, `operator`, list of referenced condition IDs in `operands`. |
| `not` | `id`, `operator`, one referenced condition ID in `operands`. |

Supported operators: `and`, `or`, `not`. The `not` operator requires exactly one operand.

### Temporal modifiers

Any leaf condition can be wrapped with a temporal modifier by adding an optional field:

| Modifier | Meaning |
|---|---|
| `persistence_n` | Condition holds for the declared consecutive-bar count. |
| `recency_n` | A qualifying event occurred within the declared trailing-bar count. |

`persistence_n` and `recency_n` apply only to leaves and are mutually exclusive. Each accepts a positive integer literal or an exact bare `{identifier}` reference to one round parameter:

In YAML, put the temporal modifier on its condition under `sfd.manifest.strategy.conditions`; a braced reference binds to a declared integer candidate list in `sfd.params`.

This fragment extends a manifest that produces `wilder_rsi_14`. MSQ draws the scalar `hold_bars` normally; each round resolves its own window without modifying the condition declaration. Use `recency_n: "{hold_bars}"` for the equivalent recency sweep.

Both YAML and native Python validate every value in the effective parameter domain, including context overrides, before creating run artifacts. Resolved windows must be positive `numbers.Integral` values, excluding booleans. Floats (including `3.0`), numeric strings, null, zero and negative values raise; no value is coerced. Missing references and formatted, indexed, attributed or compound placeholders also raise. Direct predicate evaluation enforces the same value checks.

Windows count rows within each prepared split; they do not carry predicate history across split boundaries. Persistence requires the full window; recency uses the available prefix. Ordinary and existing `factorize=True` runs retain the same decisions and parameter identity; temporal resolution does not add or change cache keys.

### Parameter template substitution

Column names and values can reference round parameters using `{param}` placeholders:

Column names may contain embedded braced parameter names; threshold values may be whole braced references. Both resolve from the current round before predicate calculation.

These are resolved per round from `round_params`. The corresponding indicator must be added to the manifest so the column is present in the data.

### Direct position transforms

A rule-based transform can emit its complete long/flat position directly. The strategy then needs only one threshold leaf over that output:

<!-- example: manifest-rule-execution-definition source=docs/examples/compiled-research/execution.yaml region=__file__ scenario=execution interface=compiled role=executed -->
```yaml
# region declaration
schema_version: '1.0'
metadata:
  name: book_compiled_execution
  mode: development
sfd:
  manifest:
    type: rule_based
    data_source:
      method: limen.data.HistoricalData.get_any_file
      params:
        file_path_or_url: inputs/spot_1h_20240101_20241231.parquet
    split_dates:
      train_start: '2024-01-02'
      train_end: '2024-01-03'
      val_start: '2024-01-04'
      val_end: '2024-01-05'
      test_start: '2024-01-05'
      test_end: '2024-01-06'
    backtest:
      product:
        kind: cash_spot
        instrument: BTCUSDT
        base_currency: BTC
        quote_currency: USDT
        quantity_step: 1e-05
        min_notional: 5
      source_interval_seconds: 3600
      fee_bps: 10
      slip_bps: 5
      take_profit_bps_column: volatility_take_profit_bps
      stop_loss_bps_column: volatility_stop_loss_bps
      barrier_features:
        method: limen.features.volatility_barriers.volatility_barriers
        params:
          span: '{risk_span}'
          regime_span: 24
          min_periods: 3
    reference_architecture: 
      limen.sfd.reference_architecture.rule_based.rule_based
    strategy:
      conditions:
      - id: entry
        name: Recorded close above open
        type: relative
        column: close
        operator: '>'
        other_column: open
      entry: entry
  params:
    risk_span:
    - 6
    - 12
uel:
  output_path: compiled-execution
  n_permutations: 2
  search_strategy:
    type: grid
  output_format: csv
  checkpoint_interval: 1
  prep_each_round: true
  recording:
    predictions: true
    alignment: true
    trade_ledger: true
    fold_details: true
# endregion declaration
```

This is the shape used by the bundled `dollar_bar_crash_reversal` template. `RuleBasedStrategy` evaluates the position with the declared costs and reports `pnl_per_trade_bps_{split}` with its `num_executed_trades_{split}` denominator in addition to the generic snapshot metrics.

### Restrictions

Rule-based YAML rejects `target` and `bar_formation` whenever either key is present, including null. The compiler would otherwise discard them. This admission also applies to saved references reconstructed by resume, Trainer, cohorts and workers. Native rule manifests still support target generation and explicitly active bar formation. Other ML-only YAML fields retain their existing warnings pending the separate closed-block admission repair.

Rule-based manifests cannot use scalers or feature ablation. Both will raise `ValueError` at data-preparation time:

- `set_scaler()` / `set_scaler_from_params()` — predicates depend on original indicator scales; scaling RSI to `[0,1]` makes threshold comparisons like `< 30` meaningless.
- `set_feature_ablation()` — predicate columns are derived from specific indicator columns and cannot be randomly dropped.

### What `prepare_data()` produces for rule-based

The rule-based path produces a different `data_dict` than the ML path:

| Prepared key | Type and responsibility |
|---|---|
| `train`, `val`, `test` | Full prepared split frames, with predicate columns. |
| `strategy` | Resolved condition definitions and entry condition ID. |
| `_alignment` | Retained source-row alignment and test bounds. |

Predicate boolean columns are added to each split DataFrame before the dict is assembled. The model receives the full DataFrames and applies the boolean logic tree at evaluation time.

### `sql_expr` escape hatch

The `sql_expr` predicate type parses a SQL expression string via `pl.sql_expr()`. Column names are referenced directly without wrappers:

| Predicate field | Meaning |
|---|---|
| `type` | `sql_expr` |
| `expr` | Polars SQL expression over declared columns; braced round parameters resolve before parsing. |

This is an expression surface parsed by Polars, not a research-code sandbox. Treat the complete Manifest and its importable implementations as code in the operator's trust domain. Parameter substitution works identically to other predicate types using `{param}` placeholders.

## Model Configuration

### `with_reference_architecture(architecture_function)`

Configure the final model function.

<!-- reference: manifest-architecture source=limen/experiment/manifest_core.py region=signature:Manifest.with_reference_architecture role=signature -->
```python
def with_reference_architecture(self: _TManifest, architecture_function: Callable[..., dict[str, Any]]) -> _TManifest:
    ...
```

The model function must accept the prepared `data_dict` as its first argument.

All remaining parameters are auto-mapped from `round_params` by signature inspection. For a model function with this signature:

<!-- reference: manifest-logreg-wrapper source=limen/sfd/reference_architecture/logreg_binary.py region=signature:logreg_binary role=signature -->
```python
def logreg_binary(data: dict[str, Any], solver: str='lbfgs', penalty: str='l2', dual: bool=False, tol: float=0.0001, C: float=1.0, fit_intercept: bool=True, intercept_scaling: float=1, class_weight: float | str | dict[int, float] | None=None, random_state: int=42, max_iter: int=100, multi_class: str='deprecated', verbose: int=0, warm_start: bool=False, n_jobs: int | None=None, l1_ratio: float | None=None, prediction_calibration_config: 'CalibrationConfig | None'=None) -> dict[str, Any]:
    ...
```

Manifest execution maps matching round keys to the current wrapper signature, using its defaults where no round value is supplied.

The built-in `logreg_binary` SFD uses this same path for the full sklearn logistic-regression constructor surface, so model tuning stays in `params()` rather than hidden inside the architecture.

Required model parameters with no defaults must be present in the round params or Limen will raise.

## Calibration

### `with_calibration()`

Opens a `CalibrationBuilder` for configuring probability calibration and threshold optimization. Call `.done()` to finalize and return to the manifest.

<!-- reference: manifest-calibration-probability_calibration source=limen/experiment/manifest_core.py region=signature:CalibrationBuilder.probability_calibration role=signature -->
```python
def probability_calibration(self, func: CalibratorProtocol, **params: Any) -> 'CalibrationBuilder':
    ...
```

<!-- reference: manifest-calibration-threshold_function source=limen/experiment/manifest_core.py region=signature:CalibrationBuilder.threshold_function role=signature -->
```python
def threshold_function(self, func: ThresholdOptimizerProtocol, **params: Any) -> 'CalibrationBuilder':
    ...
```

<!-- reference: manifest-calibration-done source=limen/experiment/manifest_core.py region=signature:CalibrationBuilder.done role=signature -->
```python
def done(self) -> 'MLManifest':
    ...
```

### `CalibrationBuilder.probability_calibration(func, **params)`

Configure the probability calibration step. `func` receives `(clf, x_val, y_val, **params)` and must return a fitted object with `predict_proba()`. Extra keyword arguments are forwarded to `func`; string values matching `round_params` keys are substituted at runtime.

### `CalibrationBuilder.threshold_function(func, **params)`

Configure the threshold optimization step. `func` receives `(y_val, val_proba, **params)` and must return `(threshold, score)`. Extra keyword arguments are forwarded the same way.

### `CalibrationBuilder.done()`

Finalises the config and returns the manifest. Raises `ValueError` if neither `probability_calibration()` nor `threshold_function()` was called first.

### Four modes

The two optional steps give a 2×2 grid of calibration modes, controlled by per-round flags:

| `use_calibration` | `use_threshold` | Behavior |
|---|---|---|
| True | True | calibrate → optimize threshold |
| True | False | calibrate → decide at 0.5 |
| False | True | raw probabilities → optimize threshold |
| False | False | no calibration injected |

Both flags default to `True`. Add `'use_calibration': [True, False]` and `'use_threshold': [True, False]` to `params()` to compare all four modes within a single experiment.

### String param resolution

Calibration params support the same resolution convention as the rest of the manifest. String values are looked up from `round_params` at runtime; non-string values pass through unchanged:

String calibration settings resolve through matching round keys; callable settings, such as the threshold metric, pass through unchanged. The [Compiled calibration case](Calibration.md) exercises all four flag combinations.

See [Calibration](Calibration.md) for the full reference including custom calibrators and threshold optimizers.

## Feature Ablation

### `set_feature_ablation(drop_count_key='feature_drop_count', seed_key='feature_drop_seed')`

Configure deterministic drop-N feature ablation after target construction.

<!-- reference: manifest-ablation source=limen/experiment/manifest_core.py region=signature:MLManifest.set_feature_ablation role=signature -->
```python
def set_feature_ablation(self, drop_count_key: str='feature_drop_count', seed_key: str='feature_drop_seed') -> 'MLManifest':
    ...
```

Declare drop-count and seed candidate lists in the SFD parameter domain. The canonical native preparation case above compares zero and one drop at seed 7.

This makes feature robustness part of the search itself.

## Feature Shims

`MLManifest.set_feature_shims(shims)` applies explicit transformations to selected numeric feature or indicator output columns, after their creation and before feature ablation, scaling and PCA. Selected values replace their raw model-input values; original indicator/feature APIs and unselected columns remain unchanged. The declaration is optional and applies to every ordinary MSQ round.

<!-- reference: manifest-shims source=limen/experiment/_column_shims.py region=signature:set_feature_shims role=signature -->
```python
def set_feature_shims(self: _TManifest, shims: list[dict[str, object]]) -> _TManifest:
    ...
```

The YAML equivalent belongs to `sfd.manifest`:

The YAML declaration lives at `sfd.manifest.feature_shims`: a sequence of mappings with selected `columns`, `method` and its method-specific fields. Column templates and braced scalar settings bind to declared round parameters.

These columns must be produced by the configured features/indicators. Column-name templates resolve from that same round. Shim scalar settings may reference existing `sfd.params`; for example `shim_d: [0.2, 0.4, 0.6]` sweeps three declared differentiation orders. The column declaration itself stays manifest configuration. MSQ uses its existing search, parameter identity and feedback interfaces.

| Method | Operation | Additional fields |
|---|---|---|
| `ratio` | `x / reference - 1` | required `reference` column |
| `scale` | `x / reference` | required `reference` column |
| `log1p` | `log(1 + x)` | none |
| `fractional_diff` | existing fixed-width fractional difference | `d` in `[0,1]` or `auto`; `threshold` defaults to `0.001`; `significance_level` defaults to `0.05` |

All ratio/scale denominators use the original frame, even when another shim replaces that reference column. Missing, nonnumeric, duplicate or overlapping selections, datetime/target/private source identity, nonfinite values, zero denominators and invalid logarithm domains raise. Temporal methods preserve leading warmup nulls and chronological row identity; they can increase the existing warmup requirement.

For `d: auto` (also the default fractional-difference order), fit each selected column on the current training partition using the bounded existing grid from zero through one. Verify the selected order with ADF and raise if none qualifies. Validation, test and Sensor use the frozen training state; walk-forward fits each fold separately. Existing Trainer reconstruction uses the persisted YAML/training partition and passes fitted state to Sensor. No new run artifact is introduced. ADF evidence belongs to its training sample: these methods provide causal transformations, not a universal stationarity or performance guarantee.

Conditional direction/sizing refits automatic shims on each earlier chronological prefix before fitting its scaler and PCA. Fold boundaries use original feature-valid source rows; inner warmup exclusions and completed-label supervision are joined by row identity. Rows without an available out-of-fold direction cannot train sizing. Final validation, test and Sensor preparation retain the frozen outer training state. Fixed-order and stateless shims retain their existing transformations. Cumulative indicators require the same history origin for replay; a shim transforms their supplied values and does not reconcile origin resets.

Ordinary sequential preparation remains fresh for each permutation. The current `factorize=True` path rejects this configuration; composing shims with factorized reuse is outside the admitted boundary.

## Data Dict Extension

### `add_to_data_dict(func)`

Append custom entries to the finalized `data_dict`.

<!-- reference: manifest-extension source=limen/experiment/manifest_core.py region=signature:MLManifest.add_to_data_dict role=signature -->
```python
def add_to_data_dict(self, func: Callable[..., dict[str, Any]]) -> 'MLManifest':
    ...
```

Use this when the model needs additional structured inputs that are not part of Limen's standard `x_*` / `y_*` schema.

## Manifest Overrides

### `with_params_override(**overrides)`

Create a deep-copied manifest with selected overrides.

<!-- reference: manifest-override source=limen/experiment/manifest_core.py region=signature:Manifest.with_params_override role=signature -->
```python
def with_params_override(self, **overrides: Any) -> 'Manifest':
    ...
```

This supports caller-owned variants, including an explicit change from train/validation/test mode to all-data training. It is not part of `Trainer.train()`, which preserves the original manifest configuration for metric reconstruction.

Supported override behavior today:

- `split_config=(1, 0, 0)` overrides the split ratios
- split overrides require a positive training ratio; validation and test ratios may be zero
- other keys are interpreted as data-source parameter overrides and are validated against the configured data-source method signature

Example:

Use `split_config` for a derived train/validation/test ratio, or an admitted source-method parameter such as `row_count_limit` for a different data selection. Neither modifies the source Manifest.

The original manifest remains unchanged.

Source overrides use the configured callable's signature, including partial keyword
parameters. Unknown names retain the accepted-name diagnostic; callable instances
and partials use their type name when they have no function name. Signature and
deepcopy failures propagate without changing the original configuration.

For ML YAML manifests, `sfd.manifest.params_override` applies accepted data-source parameter overrides to the compiled manifest. Source parameter names are checked during compilation; source constraints still apply. For `HistoricalData.get_spot_klines`, `row_count_limit` and its `n_rows` alias cannot be combined with both split-derived date limits. Reserved date-limit overrides can replace those fetch limits without updating `split_dates`. Recompiling saved YAML applies previously ignored overrides, including when resuming or reconstructing an existing experiment.

## What `prepare_data()` Produces

The manifest ultimately builds Limen's standard `data_dict`.

The foundational logistic-regression manifest produces:

- `x_train`, `y_train`
- `x_val`, `y_val`
- `x_test`, `y_test`
- `price_data_for_backtest`
- `_alignment`
- `_feature_names`
- fitted objects such as `_scaler` and fitted target parameters

`price_data_for_backtest` contains the aligned test-window OHLC data used by Limen's backtest layer.

## Function Contracts

### Indicator and feature functions

Indicator and feature functions are applied during the feature stage over a Polars lazy pipeline.

The practical contract is:

- input: frame with the columns the transform expects
- output: frame with the new columns added
- behavior: deterministic with respect to the resolved parameters

Custom manifest transforms should match Limen's built-ins: accept a frame plus keyword arguments, and return the transformed frame.

### Fitted parameter computation functions

A fitted-parameter compute function is called only on the training split.

The contract is:

- input: eager `pl.DataFrame` plus resolved kwargs
- output: one fitted value

That fitted value is then stored under the fitted-parameter name and can be referenced later in the target or scaler step.

### Target transform functions

Target transforms operate on eager `pl.DataFrame` objects after the feature stage.

The contract is:

- input: eager `pl.DataFrame` plus resolved kwargs
- output: transformed `pl.DataFrame`

Use fitted transforms when the function depends on train-only fitted state. Use plain transforms otherwise.

### Model functions

The architecture function configured in `with_reference_architecture(logreg_binary)` must:

- accept the finalized `data_dict` as its first argument
- accept any searched model parameters as named kwargs
- return a results dictionary compatible with Limen metrics and logging

If the model returns `_preds`, UEL stores them in `uel.preds` and the `Log` layer can use them for post-run analysis.

## Common Recipes

### Binary target from a fitted quantile cutoff

<!-- reference: manifest-quantile-target source=limen/targets/quantile_binary.py region=signature:QuantileBinaryTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str, source_column: str, quantile: float) -> None:
    ...
```

### Search over feature families

The canonical native preparation case tags ROC features as `momentum` and the fractional feature as `price`, then activates both through `feature_groups="momentum|price"`.

### Search over scaler choice

The Compiled selection declares its scaler through `from_params: scaler_type`; its parameter domain supplies the registered `logreg` value. A different scaler candidate is a different preparation choice.

### Build an explicit all-data variant

Use the source-bound `with_params_override` method above for an explicit all-data variant; an all-training split does not provide an independent validation or test result.

This is a separate workflow from [Trainer](Trainer.md). Trainer preserves the original split to reproduce logged metrics.

## Read Next

- Continue to [Universal Experiment Loop](Universal-Experiment-Loop.md) to run a manifest-driven SFD.
- Continue to [Calibration](Calibration.md) for the full calibration reference including custom calibrators, threshold optimizers, and all four calibration modes.
- Continue to [Historical Data](Historical-Data.md) for data surfaces a manifest can reference.
- Continue to [Data Bars](Data-Bars.md) if bar formation is part of the experiment design.
- Use [Indicators](Indicators.md), [Features](Features.md), [Targets](Targets.md), [Transforms](Transforms.md), and [Scalers](Scalers.md) as the reference layer while authoring manifests.

For signed sizing, elapsed exits, recorded execution clocks and funding configuration, see [Backtest](Backtest.md#signed-exposure-elapsed-exits-and-funding).

See [Reserve final observations](Final-Holdout.md) for project research exclusion, recorded preparation bounds and the limits of local exposure history.

### Fold promotion evidence

With `uel.recording.fold_details: true`, walk-forward campaigns retain versioned original-source, selected-source and raw/retained partition witnesses for validated native inference. Serial and parallel origins share the same trial/fold evidence. Default lean recording is unchanged. See [Trainer](Trainer.md#recorded-walk-forward-folds) for the reconstruction and clock contract, and [Cohort](Cohort.md#recorded-walk-forward-members) for explicit binary fold composition.

## Admitted YAML paths

Unknown structural fields are errors, including root, `sfd`, working/committed `lineage`, function envelopes and rule conditions. CLI saved-YAML resume applies the same narrow admission before checkpoint loading or reconstruction. Direct prevalidated compilation and scientific read-back preserve their existing boundary. Callable/source/allocation argument mappings and arbitrary `sfd.params` axis names retain their existing contracts.

Fresh and saved CLI YAML admission checks declared indicator, feature, bar-formation and pre-split-selector arguments against inspectable signatures, including inactive declarations. The frame is supplied positionally; indicator/feature `group` is consumed by the wrapper. Wrapper argument collisions, unexpected keywords and provably missing required arguments are located at their declaration paths. Defaults and genuine `**kwargs` remain valid; unavailable signatures remain indeterminate. Admission executes neither sources nor transforms.

The index follows the current validator, including its conditional rules. `Required` means presence and the listed type are checked when the condition holds. `NoUnknownKeys` lists admitted children and the actual warning/error severity; it does not imply that every child is mandatory. Open `params` mappings follow the referenced callable’s argument contract. The detailed sections above define units, defaults, timing, compatibility and returned artifacts.

| Path | Rule | Condition | Contract |
| --- | --- | --- | --- |
| `metadata` | Required | `always` | `{"type": "dict"}` |
| `sfd` | Required | `always` | `{"type": "dict"}` |
| `uel` | Required | `always` | `{"type": "dict"}` |
| `metadata.name` | Required | `always` | `{"type": "str"}` |
| `metadata.mode` | Required | `always` | `{"type": "str"}` |
| `metadata.mode` | OneOf | `always` | `{"choices": ["development", "production"]}` |
| `metadata` | NoUnknownKeys | `always` | `{"known": ["author", "created_at", "description", "limen_version", "mode", "name", "tags"], "severity": "error"}` |
| `sfd.manifest` | Required | `always` | `{"type": "dict"}` |
| `sfd.params` | Required | `always` | `{"type": "dict"}` |
| `sfd.manifest.type` | Required | `always` | `{"type": "str"}` |
| `sfd.manifest.type` | OneOf | `always` | `{"choices": ["ml", "rule_based"]}` |
| `sfd.manifest.data_source` | NoUnknownKeys | `always` | `{"known": ["method", "params"], "severity": "error"}` |
| `sfd.manifest.split_dates` | NoUnknownKeys | `always` | `{"known": ["test_end", "test_predict_guard", "test_start", "train_end", "train_start", "val_end", "val_predict_guard", "val_start"], "severity": "error"}` |
| `sfd.manifest.reference_architecture` | Required | `always` | `{"type": "str"}` |
| `sfd.manifest.reference_architecture` | Resolvable | `always` | `{}` |
| `sfd.manifest.backtest` | NoUnknownKeys | `always` | `{"known": ["barrier_features", "execution_data_source", "execution_lag_seconds", "fee_bps", "flat_threshold", "funding", "initial_equity", "max_exposure", "max_holding_seconds", "max_price_gap_seconds", "notional_rate", "prediction_mode", "product", "signal_change_bps", "slip_bps", "source_interval_seconds", "source_timestamp_role", "stop_loss_bps", "stop_loss_bps_column", "take_profit_bps", "take_profit_bps_column", "timer_interval_seconds", "timer_phase_utc_seconds"], "severity": "error"}` |
| `sfd.manifest.target` | Required | `sfd.manifest.type = ml` | `{"type": "dict"}` |
| `sfd.manifest.target.name` | Required | `sfd.manifest.type = ml` | `{"type": "str"}` |
| `sfd.manifest.target.class` | Required | `sfd.manifest.type = ml` | `{"type": "str"}` |
| `sfd.manifest.target.class` | Resolvable | `sfd.manifest.type = ml` | `{}` |
| `sfd.manifest.target` | NoUnknownKeys | `sfd.manifest.type = ml` | `{"known": ["class", "fit_params", "name", "transform_params"], "severity": "error"}` |
| `sfd.manifest.indicators` | FuncList | `sfd.manifest.type = ml` | `{"known": ["func", "include_if", "params"], "severity": "error"}` |
| `sfd.manifest.features` | FuncList | `sfd.manifest.type = ml` | `{"known": ["func", "include_if", "params"], "severity": "error"}` |
| `sfd.manifest.pre_split_data_selector` | SingleFuncBlock | `sfd.manifest.type = ml` | `{"known": ["func", "params"], "severity": "error"}` |
| `sfd.manifest.bar_formation` | SingleFuncBlock | `sfd.manifest.type = ml` | `{"known": ["func", "params"], "severity": "error"}` |
| `sfd.manifest.data_dict_extension` | SingleFuncBlock | `sfd.manifest.type = ml` | `{"known": ["func", "params"], "severity": "error"}` |
| `sfd.manifest.data_dict_extension` | NoUnknownKeys | `sfd.manifest.type = ml` | `{"known": ["func"], "severity": "error"}` |
| `sfd.manifest.scaler` | NoUnknownKeys | `sfd.manifest.type = ml` | `{"known": ["class", "from_params", "params"], "severity": "error"}` |
| `sfd.manifest.feature_ablation` | NoUnknownKeys | `sfd.manifest.type = ml` | `{"known": ["drop_count_key", "seed_key"], "severity": "error"}` |
| `sfd.manifest.feature_ablation` | ParamKeyFields | `sfd.manifest.type = ml` | `{"fields": ["drop_count_key", "seed_key"]}` |
| `sfd.manifest.pca_compression` | NoUnknownKeys | `sfd.manifest.type = ml` | `{"known": ["component_prefix", "enabled_param", "n_components_param", "scaler_param_name"], "severity": "error"}` |
| `sfd.manifest.pca_compression` | ParamKeyFields | `sfd.manifest.type = ml` | `{"fields": ["component_prefix", "enabled_param", "n_components_param", "scaler_param_name"]}` |
| `sfd.manifest.calibration` | NoUnknownKeys | `sfd.manifest.type = ml` | `{"known": ["probability_calibration", "threshold_function"], "severity": "error"}` |
| `sfd.manifest.calibration.probability_calibration` | NoUnknownKeys | `sfd.manifest.type = ml` | `{"known": ["func", "params"], "severity": "error"}` |
| `sfd.manifest.calibration.threshold_function` | NoUnknownKeys | `sfd.manifest.type = ml` | `{"known": ["func", "params"], "severity": "error"}` |
| `sfd.manifest` | NoUnknownKeys | `sfd.manifest.type = ml` | `{"known": ["acceptance", "allocation", "backtest", "bar_formation", "calibration", "data_dict_extension", "data_source", "decoder_lookback", "feature_ablation", "feature_shims", "features", "indicators", "metrics_params", "objective", "params_override", "pca_compression", "pre_split_data_selector", "reference_architecture", "required_columns", "scaler", "split_dates", "split_walk_forward", "strict_mode", "target", "type"], "severity": "error"}` |
| `sfd.manifest.strategy` | Required | `sfd.manifest.type = rule_based` | `{"type": "dict"}` |
| `sfd.manifest.strategy.conditions` | Required | `sfd.manifest.type = rule_based` | `{"type": "list"}` |
| `sfd.manifest.strategy.entry` | Required | `sfd.manifest.type = rule_based` | `{"type": "str"}` |
| `sfd.manifest.strategy.conditions` | ConditionsList | `sfd.manifest.type = rule_based` | `{"known": ["column", "direction", "expr", "id", "lookback", "name", "operands", "operator", "other_column", "persistence_n", "recency_n", "type", "value"], "severity": "error"}` |
| `sfd.manifest.indicators` | FuncList | `sfd.manifest.type = rule_based` | `{"known": ["func", "include_if", "params"], "severity": "error"}` |
| `sfd.manifest.strategy` | NoUnknownKeys | `sfd.manifest.type = rule_based` | `{"known": ["conditions", "entry"], "severity": "error"}` |
| `sfd.manifest` | NoUnknownKeys | `sfd.manifest.type = rule_based` | `{"known": ["acceptance", "allocation", "backtest", "data_source", "indicators", "objective", "reference_architecture", "required_columns", "split_dates", "split_walk_forward", "strategy", "type"], "rejected": {"bar_formation": "Field 'bar_formation' is unsupported in rule_based YAML and would be discarded", "target": "Field 'target' is unsupported in rule_based YAML and would be discarded"}, "severity": "error"}` |
| `uel.n_permutations` | Required | `always` | `{"type": "int"}` |
| `uel.search_strategy.type` | OneOf | `always` | `{"choices": ["grid", "random", "sobol"]}` |
| `uel.search_strategy` | NoUnknownKeys | `always` | `{"known": ["seed", "shuffle", "type"], "severity": "error"}` |
| `uel.output_format` | OneOf | `always` | `{"choices": ["csv", "parquet"]}` |
| `uel.pruning_strategies` | PruningStrategiesSpec | `always` | `{"known": ["params", "type"], "severity": "error"}` |
| `uel` | NoUnknownKeys | `always` | `{"known": ["checkpoint_interval", "factorize", "feedback_interval", "n_permutations", "output_format", "output_path", "prep_each_round", "pruning_strategies", "record_execution", "record_model_outputs", "recording", "search_strategy", "workers"], "rejected": {"acceptance": "Declare acceptance under sfd.manifest", "feature_shims": "Declare feature_shims only under sfd.manifest", "intra_callback": "YAML does not execute intra_callback. Remove the declaration and use UniversalExperimentLoop(intra_callback=<callable(log, msq)>); see https://github.com/Vaquum/Limen/blob/main/docs/Developer/Pruning-Strategies.md#manual-intervention", "objective": "Declare the objective only at sfd.manifest.objective"}, "severity": "error"}` |
| `<root>` | NoUnknownKeys | `always` | `{"known": ["assessment", "holdout", "lineage", "metadata", "schema_version", "sfd", "uel"], "rejected": {"acceptance": "Declare acceptance under sfd.manifest", "feature_shims": "Declare feature_shims only under sfd.manifest", "objective": "Declare the objective only at sfd.manifest.objective", "recording": "Declare recording only under uel"}, "severity": "error"}` |
| `sfd` | NoUnknownKeys | `always` | `{"known": ["manifest", "params"], "rejected": {"feature_shims": "Declare feature_shims only under sfd.manifest", "recording": "Declare recording only under uel"}, "severity": "error"}` |
| `lineage` | NoUnknownKeys | `always` | `{"known": ["committed_at", "id", "parent_id"], "severity": "error"}` |

### Nested contracts

These parsers impose additional conditions beyond the field lists. Their source identities are retained in `yaml-contracts.json` and checked when the package changes. This index does not admit the unsupported `split_config` field.

| Root | Fields or contract | Authority |
| --- | --- | --- |
| `uel` | Optional field types, positive feedback/checkpoint intervals (including rejection of anchored YAML booleans), search controls and finite execution budgets. | [UelSpec](https://github.com/Vaquum/Limen/blob/main/limen/yaml/rules.py) |
| `metadata` | Optional string, tag-list, quoted date/time and package-version shapes; null means absence. | [check_metadata](https://github.com/Vaquum/Limen/blob/main/limen/yaml/_metadata_spec.py) |
| `sfd.manifest.required_columns` | Explicit nonempty list of distinct nonblank strings; omission preserves the default and null is rejected. | [column_issue](https://github.com/Vaquum/Limen/blob/main/limen/experiment/_required_columns.py) |
| `schema_version` | Required string, exactly `"1.0"`; checked before interpreting the current-schema body. | [schema_errors](https://github.com/Vaquum/Limen/blob/main/limen/yaml/_schema_version.py) |
| `sfd.manifest` | Actual round-parameter lookup locations; literal fields keep their native meaning. | [lookup_references](https://github.com/Vaquum/Limen/blob/main/limen/yaml/_parameter_refs.py) |
| `sfd.params` | Forward declarations and Boolean `include_if` domains; proven unused axes are errors, while opaque consumers remain indeterminate. | [ParamCoverage](https://github.com/Vaquum/Limen/blob/main/limen/yaml/rules.py) |
| `sfd.manifest.strict_mode` | Optional Boolean; explicit null is invalid. | [BooleanControlsSpec](https://github.com/Vaquum/Limen/blob/main/limen/yaml/_boolean_spec.py) |
| `sfd.params.use_calibration[]` | Boolean candidates on ML manifests; omitted control defaults to true. | [BooleanControlsSpec](https://github.com/Vaquum/Limen/blob/main/limen/yaml/_boolean_spec.py) |
| `sfd.params.use_threshold[]` | Boolean candidates on ML manifests; omitted control defaults to true. | [BooleanControlsSpec](https://github.com/Vaquum/Limen/blob/main/limen/yaml/_boolean_spec.py) |
| `sfd.manifest.split_walk_forward` | `n_folds`: `int`, required; `test_bars`: `int`, required; `purge_bars`: `int`, required; `embargo_bars`: `int`, required; `anchored`: `bool`, required | [WalkForwardConfig](https://github.com/Vaquum/Limen/blob/main/limen/experiment/_walk_forward_split.py) |
| `sfd.manifest.backtest.product` | `kind`: `Literal['cash_spot', 'linear_perpetual']`, required; `instrument`: `str`, required; `base_currency`: `str`, required; `quote_currency`: `str`, required; `quantity_step`: `float \| str`, required; `min_notional`: `float \| str`, required; `initial_margin_fraction`: `float \| str`, default `1.0`; `maintenance_margin_fraction`: `float \| str`, default `0.0` | [ProductConfig](https://github.com/Vaquum/Limen/blob/main/limen/experiment/_resolve_trade_policy.py) |
| `sfd.manifest.backtest.funding` | `adapter`: `FundingAdapter`, default `funding_adapter`; `adapter_version`: `str`, default `ADAPTER_VERSION`; `preset`: `str \| None`, default `None`; `params`: `dict[str, object]`, default `field(default_factory=dict[str, object])`; `data_source`: `SourceConfig \| None`, default `None` | [FundingConfig](https://github.com/Vaquum/Limen/blob/main/limen/experiment/_resolve_trade_policy.py) |
| `sfd.manifest.allocation` | `method`: `Callable[..., dict[str, object]]`, required; `params`: `Mapping[str, object]`, required | [AllocationConfig](https://github.com/Vaquum/Limen/blob/main/limen/experiment/_allocation.py) |
| `sfd.manifest.feature_shims[]` | `columns`, `d`, `method`, `reference`, `significance_level`, `threshold` | [_FIELDS](https://github.com/Vaquum/Limen/blob/main/limen/experiment/_column_shims.py) |
| `sfd.manifest.objective` | Conditional validation in `_read_objective`; see its subject section above. | [_read_objective](https://github.com/Vaquum/Limen/blob/main/limen/yaml/_objective_spec.py) |
| `sfd.manifest.acceptance` | Conditional validation in `read_acceptance`; see its subject section above. | [read_acceptance](https://github.com/Vaquum/Limen/blob/main/limen/experiment/_acceptance_settings.py) |
| `sfd.manifest.strategy.conditions[]` | Conditional validation in `ConditionsList`; see its subject section above. | [ConditionsList](https://github.com/Vaquum/Limen/blob/main/limen/yaml/rules.py) |
| `uel.intra_callback` | Unsupported YAML key; use the native callable API. | [callback_error](https://github.com/Vaquum/Limen/blob/main/limen/yaml/_callback_spec.py) |
| `uel.pruning_strategies[]` | Closed reducer envelopes; argument mappings retain the reducer constructor contract. | [PruningStrategiesSpec](https://github.com/Vaquum/Limen/blob/main/limen/yaml/rules.py) |
| `sfd.manifest.indicators[]` | Function-envelope keys shared by indicator/feature lists and single-function stages. | [_check_func_block](https://github.com/Vaquum/Limen/blob/main/limen/yaml/rules.py) |
| `uel.recording` | Conditional validation in `parse_recording`; see its subject section above. | [parse_recording](https://github.com/Vaquum/Limen/blob/main/limen/experiment/_recording.py) |
| `assessment` | Conditional validation in `assessment_spec`; see its subject section above. | [assessment_spec](https://github.com/Vaquum/Limen/blob/main/limen/yaml/_assessment_spec.py) |
| `holdout` | Conditional validation in `holdout_spec`; see its subject section above. | [holdout_spec](https://github.com/Vaquum/Limen/blob/main/limen/yaml/_assessment_spec.py) |

## Recorded predicate and history evaluation

These variants compare two declared price floors on the 96 recorded hourly bars from January 18 through January 21, 2024. Threshold, relative, crossover, slope and SQL leaves combine through `and`, `or` and `not`; persistence requires two consecutive up bars, and recency retains a cross for three bars. History restarts at the assessment split. These are fixed rule demonstrations, not thresholds selected for profitability. Independent array arithmetic verifies every resulting decision in both interfaces. The SQL expression reads only the current open and close; arbitrary SQL still requires its own causal review.

### Python Research rules

<!-- example: manifest-python-rule-decisions source=docs/examples/python-research/rule_cases.py region=rule_signals scenario=search interface=python role=executed -->
```python
def evaluate_rules(raw):
    bars = raw.filter(pl.col('datetime').is_between(
        pl.datetime(2024, 1, 18, time_zone='UTC'), pl.datetime(2024, 1, 22, time_zone='UTC'), closed='left'))
    up = pl.col('close') > pl.col('open')
    cross = up & (pl.col('close').shift(1) <= pl.col('open').shift(1))
    persistent = (up.cast(pl.Int8).rolling_sum(2, min_samples=2) == 2).fill_null(False)
    recent = (cross.fill_null(False).cast(pl.Int8).rolling_sum(3, min_samples=1) >= 1)
    rising = (pl.col('close') > pl.col('close').shift(3)).fill_null(False)
    rows = []
    for floor in (40000., 41000.):
        signal = bars.select(((pl.col('close') >= floor) & (persistent | recent) & rising &
                              ~(pl.col('close') < pl.col('open'))).cast(pl.Int8).alias('signal'))['signal']
        rows.append({'price_floor': floor, 'signals': signal.to_list(), 'positive_decisions': int(signal.sum())})
    return {'rows': rows, 'assessment_rows': bars.height, 'history_resets_at_split': True}
```

<!-- result: manifest-python-rule-results interface=python scenario=search fields=rules.rows[].price_floor,rules.rows[].positive_decisions -->

### Compiled Research rules

The complete declaration runs with the project's CLI campaign; it requires no Python research driver.

<!-- example: manifest-compiled-rule-decisions source=docs/examples/compiled-research/rule_contracts.yaml region=declaration scenario=search interface=compiled role=executed -->
```yaml
schema_version: '1.0'
metadata:
  name: book_rule_contracts
  mode: development
sfd:
  manifest:
    type: rule_based
    data_source:
      method: limen.data.HistoricalData.get_any_file
      params:
        file_path_or_url: inputs/spot_1h_20240101_20241231.parquet
    split_dates:
      train_start: '2024-01-01'
      train_end: '2024-01-15'
      val_start: '2024-01-15'
      val_end: '2024-01-18'
      test_start: '2024-01-18'
      test_end: '2024-01-22'
    strategy:
      conditions:
      - {id: floor, name: Close above the declared floor, type: threshold, column: close, operator: '>=', value: '{price_floor}'}
      - {id: persistent_up, name: Two up bars, type: relative, column: close, operator: '>', other_column: open, persistence_n: 2}
      - {id: recent_cross, name: Recent close-open cross, type: crossover, column: close, other_column: open, direction: above, recency_n: 3}
      - {id: rising, name: Close above three bars ago, type: slope, column: close, direction: rising, lookback: 3}
      - {id: down, name: Current down bar, type: sql_expr, expr: 'close < open'}
      - {id: not_down, name: Exclude down bars, operator: not, operands: [down]}
      - {id: activity, name: Persistent or recent activity, operator: or, operands: [persistent_up, recent_cross]}
      - {id: entry, name: Combine the entry conditions, operator: and, operands: [floor, activity, rising, not_down]}
      entry: entry
    reference_architecture: limen.sfd.reference_architecture.rule_based
    backtest:
      fee_bps: 7.0
      slip_bps: 3.0
  params:
    price_floor: [40000.0, 41000.0]
uel:
  output_path: compiled-rule-contracts
  n_permutations: 2
  search_strategy: {type: grid}
  output_format: csv
  recording:
    predictions: true
    alignment: true
    trade_ledger: true
    fold_details: true
```

<!-- result: manifest-compiled-rule-results interface=compiled scenario=search fields=rules.rows[].price_floor,rules.rows[].positive_decisions -->
