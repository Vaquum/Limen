# Run and retain an experiment

The Universal Experiment Loop (UEL) runs the research definition one candidate at a time. The [Python project](examples/python-research/README.md) supplies its preparation, fitted model and score as ordinary functions. The [Compiled project](examples/compiled-research/README.md) supplies a Manifest through YAML. Both use a search strategy and the Mutable Search Queue (MSQ): the strategy generates candidates, the queue chooses what runs next, and UEL prepares and evaluates it.

This page covers:

- how CLI execution maps to UEL
- what `UniversalExperimentLoop` stores after a direct Python run
- how to run a Python SFD with a built-in search strategy
- how to enable persistence, resume, and adaptive feedback
- which runtime rules matter for manifest-driven and custom SFDs

## Prerequisites

- the project and recorded-input bundle, installed as described in either project README
- a validated YAML manifest for the CLI path, or an SFD plus compatible data for direct Python use
- a `SearchStrategy` for Python/MSQ execution
- an `experiment_dir` when you need persisted round artifacts or checkpoints

## Run the Compiled declaration

From the recorded-input bundle root, run the Compiled project directly:

<!-- example: uel-compiled-yaml source=docs/examples/commands.json region=compiled-yaml scenario=selection interface=compiled role=executed -->
```bash
limen run --no-progress-bar compiled-research/selection.yaml
```

The YAML definition and complete working-directory instructions live in the [Compiled project](examples/compiled-research/README.md). Validation, profiling and dry-run controls are documented in the [CLI reference](Command-Line-Interface.md).

`limen run` constructs UEL with a compiled SFD, a concrete search strategy, an `experiment_dir`, and the parsed YAML stored as `yaml_reference` in `metadata.json`. The result directory contains the copied manifest, `metadata.json`, `results.csv`, and `round_data.jsonl.gz`.

For trials evaluated across successive test windows, see [walk-forward sweeps](Experiment-Manifest.md#walk-forward-sweeps). Each trial runs all configured folds before its aggregate result enters the existing feedback loop.

## Direct Python run

Use Python when the experiment is expressed as an SFD, when preparation or models need custom code, or when you need direct search control. A custom SFD exposes `params()`, `prep()`, and `model()` and receives `data=` explicitly; it does not need a YAML manifest or a built-in model architecture.

For recorded execution-cost stress with a caller model, frozen event inputs and
the same pinned calibration reused in a later experiment, use the runnable
[Python cost-comparison workflow](Backtest.md#frozen-candidates-in-custom-python-research).

Pass `search_strategy` explicitly to select MSQ. The constructor still defaults to `None`, which selects the supported [standard execution contract](#legacy-execution-reference).

The Python project runs grid, seeded random, Sobol and shuffled traversal over the same caller-owned SFD. The complete function uses the recorded-data and SFD helpers from its project; no built-in model is required.

<!-- example: uel-python-search source=docs/examples/python-research/scenarios.py region=search scenario=search interface=python role=executed -->
```python
def search(data_dir, output):
    bars = hourly(data_dir, 1440)
    space = {'window': [4, 12], 'alpha': [0.1, 1.0]}
    runs = {}
    for name in ('grid', 'random', 'sobol', 'shuffled'):
        domain = ParamDomain(space)
        strategy = (GridStrategy(domain, shuffle=True, seed=42) if name == 'shuffled'
                    else {'grid': GridStrategy, 'random': RandomStrategy, 'sobol': SobolStrategy}[name](domain, **({} if name == 'grid' else {'seed': 42})))

        def model(data, round_params):
            fitted = project.fit(data['train'], round_params)
            return {'validation_net_return': project.backtest(data['validation'], fitted.predict(data['validation'].select('momentum')))}

        loop = UniversalExperimentLoop(data=bars, sfd=SimpleNamespace(params=lambda: space, prep=project.prep, model=model), search_strategy=strategy)
        loop.run(str(output / name), n_permutations=4, workers=1, progress_bar=False)
        runs[name] = project.completed_log(loop).select('window', 'alpha', 'validation_net_return').to_dicts()
    return {'runs': runs, 'common_baselines': common_baselines(data_dir, output / 'common-baselines'), 'preparation': preparation(data_dir, output / 'preparation'), 'seed': 42, 'adaptive': adaptive_search(data_dir, output / 'adaptive'),
            'feedback': mixed_feedback(data_dir, output / 'feedback'), 'helper_diagnostics': preparation_helpers(bars, data_dir, output / 'helpers'), 'target_diagnostics': label_diagnostics(bars, data_dir, output / 'targets'), 'analysis': analyze_candidates(bars, output / 'analysis'), 'input_formats': read_source_formats(data_dir), 'multiclass': evaluate_three_classes(bars), 'rules': evaluate_rules(bars)}
```

Result availability follows the selected interface and retention policy:

With native compatible retained results, UEL exposes `experiment_log`, `experiment_confusion_metrics` and `experiment_backtest_results`. The first is always populated; the latter two require post-processing and their native evidence.

Without `post_processing=True`, UEL still populates `uel.experiment_log`, but `uel._log`, `uel.experiment_confusion_metrics`, and `uel.experiment_backtest_results` remain unset. With no `experiment_dir`, the run writes `<experiment_name>.csv`; it does not persist round data or checkpoints.

The function above uses `GridStrategy` for its exhaustive cases over the same `ParamDomain`. The random-search seed controls candidate generation, not randomness inside preparation or model code. See [Advanced Search](Advanced-Search.md) for strategy choices and optional feedback.

The [convergence warning report](#convergence-warning-report) is available after run conclusion independently of `post_processing`.

Post-processing retains:

- `uel.experiment_log` with one row per round
- `uel.experiment_confusion_metrics` with one row per round
- `uel.experiment_backtest_results` with one row per round
- `uel.preds`, `uel.round_params`, and `uel._alignment` for round-level reconstruction

### Promote caller-owned fitted state

Promote evaluated fitted models with `Sensor.from_model`, then compose them with
`Cohort.from_python(selected_sensors)`. Use the [complete Python project](examples/python-research/README.md), its matching wheel and recorded-input bundle. The project includes the imports, preparation and inference helpers used by the selection function below. From the bundle root, run:

<!-- example: uel-python-selection-command source=docs/examples/commands.json region=python-selection scenario=selection interface=python role=executed -->
```bash
python python-research/research.py --scenario selection --data-dir inputs --output-dir results/python
```

The project instructions also give the working directory, input path and output directory for a repository checkout. Recorded fixtures are not installed inside the wheel.

The caller defines one-bar trades: a decision at bar `t` enters at the next open
and exits at that bar's close, with a declared 2 bp total round-trip cost. Search
sees only the first 1,440 bars. It selects the best validation return per window,
breaking ties by alpha, then reuses those exact evaluated models. The native
Cohort averages their next-bar return forecasts without refitting or unit conversion.

The trial function below evaluates the four candidates and retains their fitted models. `run_research` then selects parameters and constructs Sensors from those same objects. Execute the project entry point above; the excerpt uses helpers from its complete source file.

<!-- python-sensor-example -->
<!-- example: uel-python-selection source=docs/examples/python-research/research.py region=trials scenario=selection interface=python role=executed -->
```python
def run_trials(research, experiment_name):
    space = {"window": [4, 12], "alpha": [0.1, 1.0]}
    retained = {}

    def model(data, round_params):
        fitted = fit(data["train"], round_params)
        forecasts = fitted.predict(data["validation"].select("momentum"))
        score = backtest(data["validation"], forecasts)
        retained[(round_params["window"], round_params["alpha"])] = fitted
        return {"validation_net_return": score}

    sfd = SimpleNamespace(params=lambda: space, prep=prep, model=model)
    uel = UniversalExperimentLoop(
        data=research, sfd=sfd, search_strategy=GridStrategy(ParamDomain(space))
    )
    # An absolute filename stem writes only this CSV; no persisted round artifacts.
    uel.run(experiment_name, n_permutations=4, workers=1,
            post_processing=False, progress_bar=False)
    return uel, retained
```

Each partition loses twelve warm-up rows and its final forward-label row:
947 fitting, 467 validation and 467 held-out assessment rows. Live inference uses
prior completed context, so its first held-out bar can already be actionable;
the partition backtest is not the live Sensor's decision population.

The model callback keeps its fitted objects in ordinary caller-owned memory;
UEL retains no models with `post_processing=False`. Selection and inference reuse
those objects, including fitted preprocessing. No selected model is refitted.
The model's recorded feature names select `momentum`, while preparation declares
the known 15-minute bar availability. Output IDs declare compatible units; they
perform no conversion or methodological proof. This live route adds no Trainer
replay, compiled identity or protected final-holdout evidence.

`predict(history)` repeats the same computation as `predict_all(history)` and
returns its final result. Its cost includes O(n) metadata work plus member
prediction; preserve the completed context your preparation needs. See
[Sensor](Trainer.md#sensor) for convenience and arbitrary Python construction,
and [Cohort](Cohort.md) for custom aggregation and member-input/output mapping.
Compiled artifact binding remains a separate construction path.

## Compiled construction

Compiled Research supplies the complete YAML definition to `limen run`. The CLI validates and compiles it, executes its declared candidates and retains the original declaration. `limen assess` applies its recorded validation selection and reconstructs the selected Sensors.

<!-- example: uel-compiled-selection source=docs/examples/compiled-research/run.sh region=selection scenario=selection interface=compiled role=executed -->
```bash
limen validate compiled-research/selection.yaml
limen profile compiled-research/selection.yaml
limen run --dry-run compiled-research/selection.yaml
limen run --no-progress-bar compiled-research/selection.yaml
limen assess compiled-research/selection.yaml --results results/dev/compiled-cli
```

`loop.interrupted` reports whether shutdown was requested during the run. CLI commands retain its checkpoint and exit unsuccessfully when this is true; a batch does not start another YAML file.

## Constructor contract

<!-- reference: uel-constructor source=limen/experiment/experiment_core.py region=signature:UniversalExperimentLoop.__init__ role=signature -->
```python
def __init__(self, *, data: pl.DataFrame | None=None, sfd: Any=None, search_strategy: SearchStrategy | None=None, pruning_strategies: list[PruningStrategy] | None=None, feedback_interval: int=100, checkpoint_interval: int=1000, experiment_dir: str | Path | None=None, intra_callback: Callable[[Any, MSQ], None] | None=None, test_mode: bool=False, yaml_reference: dict[str, Any] | None=None, project_root: str | Path | None=None) -> None:
    ...
```

### Core constructor arguments

| Argument | Meaning |
|---|---|
| `sfd` | required SFD module |
| `data` | optional input dataframe; required for custom SFDs, optional for manifest-driven SFDs |
| `search_strategy` | candidate-generation strategy; selects the MSQ execution path |
| `experiment_dir` | optional directory for saved results, round data, metadata, and checkpoints |
| `pruning_strategies`, `feedback_interval`, `checkpoint_interval`, `intra_callback` | optional feedback and checkpoint controls |
| `yaml_reference` | optional parsed YAML dict stored verbatim in `metadata.json` for reproducibility |

### Data behavior

- If the SFD exposes `manifest()` and `data=` is omitted, UEL fetches data from the manifest.
- If the SFD is custom and has no manifest, `data=` is required.
- For manifest-driven SFDs, the data source used is `fetch_data()` by default; pass `test_mode=True` to use the test data source.

## `run()` contract

<!-- reference: uel-run source=limen/experiment/experiment_core.py region=signature:UniversalExperimentLoop.run role=signature -->
```python
@_holdout.run_guard
def run(self, experiment_name: str, n_permutations: int=10000, prep_each_round: bool=False, random_search: bool=True, maintain_details_in_params: bool=False, context_params: dict[str, object] | None=None, params: Callable[[], dict[str, object]] | None=None, prep: Callable[..., dict[str, object]] | None=None, model: Callable[..., dict[str, object]] | None=None, resume: bool=False, post_processing: bool=False, progress_bar: bool=True, record_execution: bool=False, record_model_outputs: bool=False, factorize: bool=False, workers: int | None=None, recording: Mapping[str, bool] | None=None) -> None:
    ...
```

### Core run arguments

| Argument | Meaning |
|---|---|
| `experiment_name` | run name; without `experiment_dir`, writes `<experiment_name>.csv`; with it, writes `results.csv` there |
| `n_permutations` | positive integer number of rounds to execute; YAML validation rejects bool, zero, negative, and values larger than the available parameter space |
| `context_params` | extra static keys injected into every round |
| `resume` | restore a checkpoint from `experiment_dir` |
| `post_processing` | compute terminal post-run metrics (`uel._log`, confusion metrics, backtest results) |
| `progress_bar` | render the experiment progress bar; on by default, disable for headless runs |
| `record_execution` | persist test snapshot series and unscaled market returns; off by default; requires `search_strategy` and `experiment_dir` |
| `record_model_outputs` | persist test probabilities and boosting iteration counts; off by default; requires `search_strategy` and `experiment_dir` |
| `recording` | choose saved round-log payload groups; all disabled by default; explicit policies require `search_strategy` and `experiment_dir`; see [recording policy](#recording-policy) |
| `factorize` | opt-in reuse of fitted preparation state across model variations, and predictions across independent snapshot economics, in a deterministic persisted MSQ grid; default `false` |

### Factorized snapshot grids

Set `uel.factorize: true` in YAML or `run(..., factorize=True)`. The existing grid still emits one ordinary result row per full parameter combination, with the same backtest executor and numeric results. This single switch enables two reuse boundaries:

- Across model variations with identical preparation inputs, reuse the fitted target and, when configured, the built-in `RobustScaler` statistics. Generate features and apply transformations again; fit and predict each distinct model afresh.
- Across independent backtest-only variations, reuse the resulting predictions and recording evidence. Run the existing snapshot executor for every economic combination.

Omitted or `false` retains fresh preparation, model fitting and prediction for every permutation, including repeated identical parameters. Default execution remains sequential. Factorization changes which computations repeat, not the experiment's round order or full-permutation identities.

Unshuffled and seeded version-2 shuffled grids support declared deterministic snapshot pipelines. Native predicates admit fixed-Cholesky Ridge, deterministic full-sample LightGBM, configured CPU XGBoost, supported logistic solvers, DLinear and rules. Numerical work runs under one native thread. Reuse keeps the original continuous predictions and uses their positive direction only for snapshot execution. Economic fields are recomputed for every fee/slippage/notional/barrier candidate.

Deterministic exported numerical transforms, targets and scalers share the same admission boundary. A random target requires an explicit integer seed. Registry-selected scalers, conditional transforms, PCA and active calibration can participate when their actual components are declared and independent of economic axes. Active calibration settings bound through architecture partials retain the same economic-dependency checks as manifest configuration. Only the previously proved target/RobustScaler path reuses fitted preparation across different model keys; other admitted paths prepare afresh per signal key. Prepared feature matrices and fitted models are not retained automatically.

Pruning, loop callbacks, overridden pipelines, event execution, objectives, allocation, walk-forward and multiple workers remain outside this snapshot method. Missing active declarations, unsafe settings, economic dependencies, interventions and candidates outside the admitted domain fail explicitly. Inactive components need no declaration. Initial admission failures precede UEL artifact writes; observed invalid output or replay failures preserve the already committed successful prefix.

#### Reuse declarations

A caller attaches `declare_reuse(component, ReuseContract(...))` to its feature, target, scaler, selector, extension or architecture. Local callables and validators work in this invocation; no worker transport is involved.

<!-- reference: uel-reuse-contract source=limen/experiment/_factorized_contract.py region=signature:ReuseContract role=signature -->
```python
@dataclass(frozen=True)
class ReuseContract:
    validate: Callable[[Mapping[str, object]], None]
    parameter_keys: tuple[str, ...] = ()
    validation_keys: tuple[str, ...] | None = None
    consumes: tuple[str, ...] | None = None
    prediction_kind: Literal['binary', 'continuous'] | None = None
    replay: Callable[[Mapping[str, object], Mapping[str, object], Mapping[str, object]], dict[str, object]] | None = None
    replay_inputs: tuple[str, ...] = ()
```

<!-- reference: uel-declare-reuse source=limen/experiment/_factorized_contract.py region=signature:declare_reuse role=signature -->
```python
def declare_reuse(component: object, contract: ReuseContract) -> None:
    ...
```

`validate` receives effective settings and raises `ValueError` for an unproved combination. `parameter_keys` names required settings; `validation_keys` names every setting the validator reads, or defaults to all delivered settings. Eligibility preflight enumerates each component's joint projection with its own activation selectors, keeping a conditional transform's flag and feature group together. Fixed-signature architectures receive only their named settings; `**kwargs` architectures and data extensions can receive the full domain. Their default eligibility projection retains those joint reads; declare narrower `validation_keys` only when the validator is independent of the omitted settings. Every drawn candidate is checked again before fitting.

`consumes` names scientific input paths; its default includes every delivered setting. Declare a smaller set only when the computation is independent of the others. The author owns the truth of the determinism and dependency declarations. An ML architecture declares `prediction_kind='binary'` or `'continuous'` for native snapshot replay, or supplies `replay(scientific, data, params)` to recompute its own economic fields. `replay_inputs` names additional prepared values needed by that callback. Each callback receives an isolated copy captured before model computation; mutation cannot change later cache hits. Custom rule architectures require explicit economic replay so every partition uses their actual signals.

Native Ridge requires `solver='cholesky'`, `positive=False`, `copy_X=True`. LightGBM retains its full-row/full-column, integer-seed, deterministic single-thread requirements. Logistic regression requires `n_jobs=1`; stochastic solvers also require an integer seed. XGBoost requires CPU histogram trees, depthwise uniform full sampling, one tree per iteration, one thread and an integer seed. Unseeded models and unfrozen optional weights remain unproved.

Both caches belong to one sweep invocation and contain only successful work. Fitted artifacts are copied before use by another round. The prediction cache retains predictions, price/provenance and recording evidence; each cache retains at most **128 entries**, evicting the least recently used entry. Hits refresh recency; evicted keys recompute on their next access. Prediction entries include copied fitted-preparation evidence, so this is 128 prediction entries plus 128 separate preparation entries, not 128 fitted objects total or a byte/RSS limit. Source observations, active-round work, domain/seen-set and result/log growth remain separate costs. Reuse fits a key once while resident; shuffled prefixes may provide few hits and eviction may repeat work without changing scientific outputs. Independent runs each own these costs. Log post-processing and resumed invocations retain ordinary reconstruction work. CLI resume reads the effective setting from metadata; caches restart empty. Strict-mode failures consume a terminal candidate, retain a nonempty failure marker and have no successful prediction record; recovery joins only successful occurrences. No automatic retry or recovery before the first existing checkpoint is added. Changing YAML `factorize` changes canonical manifest identity but never full-permutation round IDs.

The recorded-data benchmark isolates the fitted-preparation extension: both campaigns use `factorize=True` and retain prediction reuse; the baseline clears only the fitted-preparation cache before each evaluation. It records actual target/scaler fits, stage timings, end-to-end elapsed time, peak process RSS and scientific-output digests. It uses fixed 2024 hourly observations, pinned splits and single-threaded LightGBM settings; its bounded workload makes no universal speedup claim.

The maintained [factorized preparation probe](../tests/benchmark_factorized_preparation.py) records this comparison separately from the canonical project's scientific equivalence assertions.

### Parameter and result names

UEL flattens parameters and evaluated results into one row. At admission it checks
both the effective parameter domain and `context_params`. A search strategy supplies
the effective domain; legacy execution uses `self.params` after a `params()` override.
Context values still take precedence over same-named domain values.

These bookkeeping names belong to UEL and cannot be supplied by either caller
parameters or a successful model result: `id`, `execution_time`, `strict_mode_error`,
`_warnings`, `_convergence_warning`, `_dropped_features`, `_param_hash`, `_id`,
`_round_index`, `_injected`, `_generation_index`, `_search_strategy`.
The objective's `val_backtest_total_return` is additionally reserved for parameters
while that objective is active; its evaluated result is valid. Legacy execution
also reserves `_experiment_details` for parameters when `maintain_details_in_params=True`.

On every successful round, UEL rejects overlap between the parameters merged into
the row and retained result keys, even when their values are equal. Model artifacts
`extras`, `models`, `_model` and `_preds` are extracted from the row and excluded from
that comparison. Parameters named `extras` or `models` remain valid. Framework-created
strict-mode failures and generated metadata retain their existing behavior.

Collisions raise `ValueError` identifying sorted names and their sources. Static
refusal happens before UEL output initialization or recovery writes. Dynamic refusal
happens before the affected row reaches return tracks, CSV/round records, the log or
feedback. Setup files and an earlier valid prefix may remain; use a fresh run after
renaming a colliding contract. Recovery also refuses committed-range round identities
without corresponding successful CSV occurrences before rewriting existing evidence.
Supported duplicate and torn uncommitted tails retain their recovery behavior.

This policy applies to legacy serial, MSQ serial (including factorized execution),
and admitted parallel UEL runs. Standalone MSQ naming rules and propagation of
nonreserved underscore context keys are unchanged. Protected runs keep the existing
research-attempt lifecycle: prior provenance is preserved, and a refused new attempt
is recorded as started then failed.

### SFD execution contract

On the ordinary MSQ path, UEL calls the SFD's preparation and model functions for each candidate. Manifest-driven SFDs supply these functions through their manifest. Custom SFDs supply them directly and require explicit `data=`. Configure candidate values in the strategy's `ParamDomain`; configure custom functions in the SFD. The `factorize` option above has a separate, narrower manifest-only contract.

When `search_strategy` is set, the legacy-only `run()` arguments `prep_each_round`, `random_search`, `maintain_details_in_params`, `params`, `prep`, and `model` are accepted but ignored. Configure the strategy and SFD instead.

For saved custom runs, the SFD must expose `__name__` and preparation must supply the expected `_alignment` metadata. Arbitrary custom SFD execution does not imply Trainer compatibility: Trainer reconstructs a manifest from a saved `yaml_reference`.

## What UEL stores after a run

Primary attributes are listed below. `uel.data`, `uel.params`, and `uel.experiment_log` are available after every successful run. `uel._log` and round artifact collections are retained only when `run(..., post_processing=True)` is enabled.

| Attribute | Meaning |
|---|---|
| `uel.data` | dataframe used by the run |
| `uel.params` | parameter space in use |
| `uel.round_params` | actual parameter values retained for each successful round when post-processing is enabled |
| `uel.experiment_log` | main round-by-round experiment log |
| `uel.convergence_report` | recorded convergence warning counts, rates and parameter-value summaries after run conclusion |
| `uel.experiment_confusion_metrics` | confusion-style analysis derived from predictions |
| `uel.experiment_backtest_results` | backtest-style analysis derived from predictions |
| `uel.preds` | test predictions retained when post-processing is enabled |
| `uel.scalers` | fitted scalers retained when post-processing is enabled |
| `uel._alignment` | alignment metadata retained when post-processing is enabled |
| `uel._log` | internal `Log` object for deeper analysis |

### Alignment metadata

Each entry in `uel._alignment` includes:

- `missing_datetimes`
- `first_test_datetime`
- `last_test_datetime`

This is what lets downstream analysis stay aligned with the actual test window seen by a round.

### Convergence warning report

After a completed standard, MSQ or walk-forward run, inspect `uel.convergence_report`:

| Field | Meaning |
|---|---|
| `convergence_warning_rounds` | Finished, observed rounds with a recorded convergence warning. |
| `observed_rounds` | Finished rounds with available category diagnostics. |
| `convergence_warning_pct` | Warning-bearing share of observed rounds, or null without observations. |

This dictionary reports explicitly recorded `ConvergenceWarning` categories, including subclasses. Warning message text is not used to infer convergence problems. A recorded absence of that warning means no warning was observed; it does not prove the model converged.

`convergence_warning_pct` is 100 times `convergence_warning_rounds` divided by `observed_rounds`: finished rounds with available diagnostics. Missing diagnostics, older records without the category observation, and strict-mode failed rounds are unavailable and excluded from that denominator. `rounds` and `unavailable_rounds` list the total and unavailable counts separately. With no observed rounds, the percentage is `None` in memory and JSON `null`.

For walk-forward runs, each trial counts once: a warning in any observed fold makes the trial warning-bearing. A trial can be recorded without a warning only when every fold has available diagnostics and none records one; otherwise its observation is unavailable.

`parameter_patterns` lists each parameter value associated with at least one recorded warning. Each entry carries `parameter`, a display string `value`, `observed_rounds`, `convergence_warning_rounds` and `convergence_warning_pct`. Its denominator includes every observed round with that value, including those without warnings. These are marginal descriptive frequencies, not evidence that a parameter caused a warning; combinations can be confounded and groups can overlap across parameters.

UEL computes this report even with `post_processing=False`. With `experiment_dir` configured, it writes `convergence_report.json` there. Otherwise it retains the dictionary in memory. Both paths log the overall counts and rate at run conclusion.

### Deeper post-run analysis

With `post_processing=True`, UEL constructs a live `Log` at run conclusion. It supports explicit retained row analysis. Captured aggregates can also be read through `limen.Log(uel_object=uel)` without that option, or through a CSV-backed `Log`; see the [canonical aggregate and row contracts](Log.md). The retained path exposes methods such as:

- `uel._log.permutation_prediction_performance(round_id=0)`
- `uel._log.permutation_confusion_metrics('price_change', round_id=0)`
- `uel.experiment_parameter_correlation('auc')`

### Inline metrics in the round log

The `confusion_*` and `backtest_*` columns in `uel.experiment_log` and `results.csv` are produced **inline** — computed once per round inside the architecture's `evaluate()` while the round runs, not by post-run processing. `evaluate(data, inline_metrics=True)` appends them to the metrics dict the model returns, and UEL merges that dict verbatim into the round row. UEL forwards no flag of its own; on the manifest-driven path the architecture wrapper pins `inline_metrics=True`, so these columns are always present.

With `post_processing=True`, `uel.experiment_backtest_results` reads captured backtest columns through `Log`, while `uel.experiment_confusion_metrics` keeps the existing explicit `price_change` row summary. The latter applies trimming and additional statistics; it is distinct from the untrimmed captured confusion view returned by `Log.experiment_confusion_metrics()` with no argument. [Log](Log.md) defines their values, units, evidence requirements and walk-forward fold behavior.

The inline confusion and backtest returns are price-gated: confusion counts always appear, but the price-derived `confusion_*_mean_return_pct` and `backtest_*` metrics are computed only when `price_data_for_backtest` is present for the round. Configured TP/SL requires valid prices and provenance before training, even when the current candidate resolves to null; unavailable prices raise. See [Reference Architecture](Reference-Architecture.md) for the `evaluate()` contract and the exact keys each mode adds.

TP/SL candidates belong to `sfd.manifest.backtest` references and ordinary search-parameter lists. Each candidate repeats preparation, training, and prediction; an exit-only sweep preserves predictions only with deterministic entry behavior. Rule-based entry architectures evaluate the fixed exits per split, while UEL still omits their post-run aggregate backtest table. Captured aggregate reads preserve the recorded execution result; explicit replay has separate source-evidence requirements. [Log](Log.md#backtest-surface) defines the aggregate contract, and [Backtest](Backtest.md#tunable-take-profit-and-stop-loss) defines the trading policy.

## Persisting Python/MSQ runs

When UEL is instantiated with a concrete `search_strategy` and an `experiment_dir`, Limen stores structured artifacts there. This path uses `results.csv` as the round log filename rather than `<experiment_name>.csv`.

| File | Meaning |
|---|---|
| `results.csv` | streaming round log; if a round fails a `strict_mode` null check, a `strict_mode_error` column records the error message and all metric columns for that round are empty |
| `round_data.jsonl.gz` | round identity and params, with payloads selected by the [recording policy](#recording-policy) and existing recording flags |
| `checkpoint.json` | checkpoint state for resumption |
| `audit.jsonl` | feedback-controller audit trail |
| `interventions.json` | optional external intervention file polled by the feedback controller when the file exists |
| `metadata.json` | experiment metadata used by `Trainer` |
| `convergence_report.json` | convergence warning report written at run conclusion, independently of post-processing |

Saved state powers checkpointing and resumability. The CLI also stores the YAML definition in `metadata.json`; [Trainer](Trainer.md) and CLI resume require that reconstructible `yaml_reference`. A Python run with arbitrary custom functions and an `experiment_dir` does not automatically meet this requirement.

### Record execution

Set `uel.record_execution: true` in YAML, or pass `record_execution=True` to a Python/MSQ run with `experiment_dir`. This is independent of `post_processing`. Each successful round gains `execution` in `round_data.jsonl.gz`: full-precision `pos`, `gross`, and `net` arrays in test-row order, each multiplied once by the resolved `notional_rate`. Positions include execution lag and exits; regressors may backtest directional signals rather than their continuous saved predictions. Rule-based strategies record only test execution.

The same flag records the sibling `market: {"ret": [...]}`: unscaled returns from the original aligned test prices, in the same order and length as `execution.pos`. Each value is `close[i] / close[i-1] - 1` since the previous retained test row; row gaps can therefore span several source bars. The first value is JSON `null`, since the evaluated window supplies no predecessor. A value is also `null` when the original open, close, close-minus-open or previous close is NaN, the previous close is zero, or the computed return is nonfinite. This uses price tradability, independently of signals, execution lag, costs, notional and TP/SL settings; it is not a finite-positive-price mask.

When no snapshot runs (missing prices, disabled inline metrics, a custom producer, or event execution), both `execution` and `market` are `null`. A flat snapshot has full-length zero execution arrays and still records market returns. Event execution retains its existing `trade_ledger` by default. With recording off, neither field is added.

Declare the required artifacts before execution. The Compiled project retains predictions, alignment, trade ledgers and fold details through YAML; its proof verifies prediction payload hashes and compares the decoded values with the original label population:

<!-- example: uel-compiled-recorded-predictions source=docs/examples/compiled-research/research.yaml region=run scenario=search interface=compiled role=executed -->
```yaml
uel:
  output_path: compiled-research
  n_permutations: 8
  workers: 1
  search_strategy:
    type: grid
    shuffle: true
    seed: 17
  prep_each_round: true
  checkpoint_interval: 1
  feedback_interval: 1
  output_format: parquet
  recording:
    predictions: true
    alignment: true
    trade_ledger: true
    fold_details: true
```

To replay snapshot summaries, construct an `ExecutionResult` from retained `pos`, `gross` and `net` arrays and apply the package-local snapshot ledger with notional `1.0`. Those arrays already contain the resolved notional; applying the original notional again would scale twice. For the shipped snapshot strategy without TP/SL, `gross[i] == execution.pos[i] * market.ret[i]` on available market rows. TP/SL uses fill-based exit returns, so a residual against original market closes includes exit effects. A timing diagnostic can subtract mean position times mean market return from mean gross return over one common, nonempty finite population; this adds no Limen metric. Net returns include costs, and compounded market return is a separate quantity.

Ordinal halves use `k = n // 2`, H1 rows `[0:k)` and H2 rows `[k:n)`; an odd middle row belongs to H2. For `n >= 2` with valid prices and no interior nulls, compounding available returns gives H1 `close[k-1] / close[0] - 1` and H2 `close[n-1] / close[k-1] - 1`. H2 includes the move across the boundary. Skipping interior nulls does not guarantee those endpoint identities. Window lengths and split dates can differ between rounds; calendar comparisons still require row identity and alignment.

Recording adds four arrays in total, including one full-precision market array per round. Serialized size and decoded-object overhead vary with the returns, test-window length and round count; Trainer and Cohort load whole round records.

Python resume must pass the same flag; CLI resume uses the effective setting saved in metadata, including Python overrides. Changing it raises before artifacts are rewritten. Older metadata without the flag means `false`. Earlier execution-only records remain unchanged when resumed; readers must treat a missing `market` as unavailable, while new snapshot records include it. No artifact migration is performed. Existing resume requirements, including complete successful round records through the checkpoint, still apply.

### Record model outputs

Set `uel.record_model_outputs: true` in YAML, or pass `record_model_outputs=True` to `run()`. Each successful round records `probs` in `round_data.jsonl.gz`, in test-row order (matching `preds` when enabled), with `optimal_threshold` and `threshold_rule`. These are the original test probabilities used for that evaluation, including calibration when enabled. Apply `>` for uncalibrated predictions and `>=` for the calibration/threshold path; the uncalibrated threshold is `0.5`. RandomBinary records its existing `0.1`/`0.9` surrogate scores, which encode its sampled predictions. One-class LightGBM fits retain their original scores and record a constant decision boundary: `0`/`>=` for class `1`, or `1`/`>` for class `0`. Architectures without probabilities record `probs: null`.

LightGBM and XGBoost add `best_iteration` to `results.csv`: the number of boosting iterations actually used for prediction. LightGBM uses its positive `best_iteration_`, otherwise `n_iter_`. Tree-based XGBoost uses its zero-based best iteration plus one, otherwise the fitted booster's round count. XGBoost `gblinear` uses the final fitted round count because it does not retain an earlier model. Counts are available with early stopping disabled and with `inline_metrics=False`.

The option is independent of `record_execution` and `post_processing`; leaving it off preserves existing outputs. Python resume must pass the same flag; CLI resume uses the effective setting saved in metadata, including Python overrides. Metadata records an enabled setting, and a changed setting rejects resume before artifacts are rewritten. Older metadata without the flag means `false`. Probability arrays increase JSONL storage with test-window length.

### Recording policy

Use `uel.recording` in YAML or `run(recording={...})` in a persisted Python/MSQ run to omit named groups from `round_data.jsonl.gz`. Every group defaults to `false`; new runs omit bar-level payloads unless explicitly enabled. An explicit all-true policy restores full recording. Values must be literal booleans: unknown keys, parameter references and non-booleans are errors. Explicit policies require both `search_strategy` and `experiment_dir`.

| Key | When `false` |
|---|---|
| `predictions` | Omit saved `preds`, including within retained folds. |
| `alignment` | Omit row-sized `missing_datetimes`, fold `test_datetimes` and contract `signal_rows`; retain dated bounds, source checksums and the full contract digest. |
| `trade_ledger` | Omit the complete event ledger; retain the trade contract, its digest and learning binding. |
| `fold_details` | Omit the entire nested `folds` payload, including fold-local results, bounds and provenance; retain the outer occurrence identity and parameters. |

Choose the evidence policy before the first candidate runs. The two recovery functions below enable all four groups explicitly before the first run. Full recording supports every recovery mode; a smaller policy may omit groups the run does not use.

The Python equivalent is `uel.run(..., recording=dict.fromkeys(("predictions", "alignment", "trade_ledger", "fold_details"), True))`. On a new run, `recording=None` uses the configured YAML policy, or all-false defaults when absent. An explicit native mapping replaces the entire YAML policy before unspecified groups default to `false`: native `{"trade_ledger": True}` overrides YAML `{"predictions": True}` with predictions disabled.

For a compiled SFD, its compiled YAML is authoritative even without a separate `yaml_reference`. A conflicting recording declaration in an explicitly supplied `yaml_reference` rejects before artifacts. New persisted recording runs retain the original compiled YAML through the existing `yaml_reference` metadata. The frozen effective policy is stored separately as `metadata.recording`. Changing YAML `uel.recording` changes the existing canonical manifest ID; a native override preserves the source YAML and its ID. Neither operation changes the manifest-ID algorithm. New persisted runs save the effective policy even when no policy was declared.

This policy filters copied serialized records only. Models, predictions used for scoring, scalar CSV/Parquet results, feedback and live `post_processing` retain their full evidence. When folds remain, the other controls apply within them. Walk-forward `trial_returns.parquet` and DSR computation are unchanged. Existing `record_execution` and `record_model_outputs` remain independent and off by default: omitted predictions can coexist with recorded execution, market returns or probabilities.

Disabling `fold_details` or `alignment` makes PBO unavailable because the saved log lacks fold/timestamp evidence. Explicit `acceptance.max_pbo` or `acceptance.pbo_blocks` requires both payloads; count-only declarations, including two, request the same evidence. Conflicts reject before run artifacts, and direct explicit requests reject incompatible saved policies before report writes. Omitted-count defaults retain unavailable PBO plus independently supported DSR. Explicit counts also require a search strategy and experiment directory. See [acceptance configuration](Experiment-Manifest.md#acceptance-thresholds) for the canonical count, persistence and regeneration rules.

Resume requires `predictions` and `alignment`, including saved `missing_datetimes`, even with `post_processing=False`. Event execution also requires `trade_ledger`; walk-forward requires `fold_details` and the applicable evidence within every fold. Ordinary single-split snapshots may disable both ledger and fold recording, including snapshots with only backtest costs. A persisted trade contract requires ledger evidence even for a custom Python SFD without a Manifest; caller-owned ledger payload alone does not establish an engine recovery contract.

Choose that policy on the initial run. Resume rejects changed policies or YAML declarations and missing required committed payloads before truncating round data, rewriting CSV/checkpoints or regenerating fold returns. Artifact references must still resolve through the ordinary integrity checks. An omitted resume policy uses saved metadata; legacy metadata without a policy means all-true recording. Default all-false runs remain non-resumable.

Supported single-split Trainer/Sensor reconstruction and built-in Cohort selection retain their identities, parameters, scalar results and required exact trade/learning bindings. Omitted row-level evidence cannot be inspected from the log; custom readers must require the fields they use. Walk-forward reconstruction requires the separate [promotion evidence](Trainer.md#recorded-walk-forward-folds). Reducing recording changes stored evidence, not live retention or model/preparation memory.

The [recorded payload inventory](https://github.com/Vaquum/Limen/blob/main/tests/fixtures/recording_payload_inventory.json) measures two production rounds per case from `spot_1h_20240101_20241231.parquet`, with both execution and model-output recording enabled for ordinary cases and execution recording enabled for walk-forward. For records `R`, logical bytes are `L(R) = sum(len((json.dumps(r) + '\n').encode('utf-8')) for r in R)`, using Python's default JSON settings. Each section's weight is `L(full) - L(without_section)`; its percentage divides that difference by `L(full)` for that case alone. Compressed sizes include the round log and shared market blobs. The alignment measurement removes the entire alignment object; the policy retains its dated bounds, so that percentage is not its omission saving.

| Case | Logical bytes | Compressed bytes | Largest payload sections |
|---|---:|---:|---|
| Rule snapshot | 17,680 | 5,987 | Execution 53.85%; market 37.69% |
| Binary snapshot | 25,376 | 9,790 | Execution 43.08%; market 26.08%; probabilities 22.67% |
| Ridge snapshot | 22,399 | 8,408 | Execution 37.49%; market 29.55%; predictions 27.91% |
| Event | 245,352 | 36,981 | Ledger 89.13%; retained trade contract 10.23% |
| Walk-forward | 31,677 | 10,311 | Fold results 34.99%; alignment 20.34%; execution 19.41% |

In the recorded full-payload baseline, omitting only the event ledger removes 218,682 of the two-event sample's 245,352 logical bytes (89.13%) and recompresses it to 4,668 bytes. With alignment recording disabled, per-signal identities are also omitted; Trainer compares reconstructed contract fields and the original full contract digest. These small fixture observations are not campaign projections or universal savings claims; defaults already omit execution/market and model-output recordings, and compressed omission savings are not additive. Stored counts describe the measured baseline, not a portable exact-byte oracle for other numerical libraries or zlib implementations.

The maintained [payload-projection regression](../tests/test_recording_policy.py) verifies structural omissions and relative storage sizes against the recorded fixture. It is separate from the project examples, which prove their actual retained outputs.

Structural and relative-size regression checks run with `python -m pytest -q tests/test_recording_policy.py -k payload_projection`; exact compressed counts are not the acceptance criterion.

## Optional adaptive search

Optional reducers can adapt the remaining search while UEL runs. Both projects request four candidates and budget the run to three; the recorded audit explains why the fourth did not execute:

### Python Research

<!-- example: uel-budget-python source=docs/examples/python-research/scenarios.py region=adaptive_search scenario=search interface=python role=executed -->
```python
def adaptive_search(data_dir, output):
    space = {'window': [4, 12], 'alpha': [.1, 1.]}

    def model(data, round_params):
        fitted = project.fit(data['train'], round_params)
        return {'validation_net_return': project.backtest(data['validation'], fitted.predict(data['validation'].select('momentum')))}

    sfd = SimpleNamespace(__name__=__name__, params=lambda: space, prep=recordable_prep, model=model)
    loop = UniversalExperimentLoop(data=hourly(data_dir, 1440), sfd=sfd, search_strategy=GridStrategy(ParamDomain(space)),
                                   experiment_dir=output, checkpoint_interval=1, feedback_interval=1,
                                   pruning_strategies=[BudgetReducer(max_permutations=3)])
    loop.run('python-adaptive-search', n_permutations=4, workers=1, progress_bar=False)
    checkpoint = json.loads((output / 'checkpoint.json').read_text())['msq_state']
    return {'rows': project.completed_log(loop).select('window', 'alpha', 'validation_net_return').to_dicts(),
            'audit': [json.loads(line) for line in (output / 'audit.jsonl').read_text().splitlines()],
            'checkpoint_yielded': checkpoint['yielded_count'], 'checkpoint_remaining': checkpoint['trim_budget']}
```

### Compiled Research

<!-- example: uel-budget-compiled source=docs/examples/compiled-research/adaptive.yaml region=budget_config scenario=search interface=compiled role=executed -->
```yaml
uel:
  output_path: compiled-adaptive
  n_permutations: 4
  search_strategy:
    type: grid
  output_format: csv
  checkpoint_interval: 1
  pruning_strategies:
  - type: budget
    params:
      max_permutations: 3
  feedback_interval: 1
```

The budget reducer can trim the remaining queue during a feedback cycle, so the number of completed rows may be lower than the requested permutation budget. `results.csv` and `round_data.jsonl.gz` track completed rounds; `audit.jsonl` records the intervention; checkpoints follow `checkpoint_interval`.

## Resume in practice

Python/MSQ runs can resume when the applicable [recording groups](#recording-policy) were enabled on the initial run, and an `experiment_dir` and a checkpoint are available:

### Resume Python Research

<!-- example: uel-recovery-python source=docs/examples/python-research/scenarios.py region=recovery scenario=recovery interface=python role=executed -->
```python
def recovery(data_dir, output):
    bars = hourly(data_dir, 1440)
    space = {'window': [4, 12], 'alpha': [0.1, 1.0]}

    def model(data, round_params):
        fitted = project.fit(data['train'], round_params)
        forecasts = fitted.predict(data['validation'].select('momentum'))
        return {'validation_net_return': project.backtest(data['validation'], forecasts), '_preds': forecasts}

    sfd = SimpleNamespace(__name__=__name__, params=lambda: space, prep=recordable_prep, model=model)

    def stop_after_two(log, msq):
        if log.height == 2:
            signal.raise_signal(signal.SIGINT)

    def loop(callback=None, directory='experiment'):
        return UniversalExperimentLoop(data=bars, sfd=sfd, search_strategy=GridStrategy(ParamDomain(space)), experiment_dir=output / directory, checkpoint_interval=1, feedback_interval=1, intra_callback=callback)

    recording: dict[str, bool] = dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True)
    interrupted = loop(stop_after_two)
    interrupted.run('python-recovery', n_permutations=4, workers=1, progress_bar=False, recording=recording)
    completed_before_resume = project.completed_log(interrupted).height
    resumed = loop()
    resumed.run('python-recovery', n_permutations=4, workers=1, progress_bar=False, recording=recording, resume=True)
    with gzip.open(output / 'experiment/round_data.jsonl.gz', 'rt') as stream:
        records = [json.loads(line) for line in stream]
    full = loop(directory='uninterrupted')
    full.run('python-recovery', n_permutations=4, workers=1, progress_bar=False, recording=recording)
    if not project.completed_log(resumed).drop('execution_time').equals(project.completed_log(full).drop('execution_time')):
        raise ValueError('Resumed Python results differ from uninterrupted research')
    return {'interrupted_rounds': completed_before_resume, 'rounds': project.completed_log(resumed).height, 'round_indices': [row['_round_index'] for row in records], 'round_ids': [row['round_id'] for row in records], 'fitted_state_restored': False, 'uninterrupted_equivalence': True}
```

### Resume Compiled Research

<!-- example: uel-recovery-compiled source=docs/examples/compiled-research/run.sh region=recovery scenario=recovery interface=compiled role=executed -->
```bash
limen run --no-progress-bar compiled-research/recovery.yaml
limen run --no-progress-bar --resume results/dev/compiled-recovery
```

The recovery tests interrupt real execution, resume its checkpoint and compare final identities and metrics with uninterrupted runs. The Compiled recovery YAML pins its local Parquet source with `expected_sha256`; initial execution and resume reject changed bytes before fitting. Python recovery restores the caller-defined queue and results.

Use the same `experiment_dir`, strategy type, and reducer configuration when resuming.

For strategy and feedback controls, continue to [Advanced Search](Advanced-Search.md) and [Reducers And Feedback](Reducers-And-Feedback.md).

## Legacy execution reference

UEL also retains the standard `ParamSpace` execution path for existing callers. Omitting `search_strategy` selects it. New examples use Python/MSQ; this reference records the legacy contract without changing its support.

| Legacy `run()` argument | Behavior |
|---|---|
| `random_search` | random sampling, or deterministic enumeration when `False` |
| `prep_each_round` | repeat preparation; `True` is required for manifest-driven SFDs; custom SFDs may prepare once when it is `False` |
| `params` | callable override for the parameter space |
| `prep`, `model` | callable overrides for custom SFDs; disallowed for manifest-driven SFDs |
| `maintain_details_in_params` | expose `_experiment_details` to the current round |

The path writes `<experiment_name>.csv`, or `<experiment_dir>/<experiment_name>.csv` when a directory is supplied. It uses integer round IDs and has no checkpoint/resume or MSQ feedback. `post_processing=True` enables the same live analysis attributes.

With `random_search=False`, combinations follow declared parameter order, with the first parameter varying fastest. Legacy UEL random sampling exposes no seed; module-global `random.seed(...)` does not pin it. The [ParamSpace utility](Utilities.md#paramspace) exposes its own seed for direct helper use. These ordering and sampling contracts differ from the MSQ strategies.

### Direct standard run

For existing callers, omit `search_strategy` and use the legacy arguments. This supported contract remains available:

Use the source-bound constructor and `run()` signatures above: omit `search_strategy`, set `prep_each_round=True` for Manifest preparation, and choose `random_search=False` for legacy enumeration. The canonical projects use explicit MSQ strategies.

## Common Errors

### Legacy manifest-driven runs require `prep_each_round=True`

A manifest-driven SFD with `prep_each_round=False` raises `prep_each_round must be True for manifest-driven SFDs`. Set `prep_each_round=True`.

### Legacy manifest-driven runs cannot override `prep` or `model`

Passing `prep=` or `model=` to `run()` for a manifest-driven SFD raises `Cannot override prep/model when SFD has manifest`. Put the logic into the manifest, or switch to the custom SFD path.

### Custom SFDs require explicit `data=`

A custom SFD with omitted `data=` raises `data parameter required for custom SFDs using custom functions approach`.

### Resuming requires a search strategy

Calling `run(resume=True)` without a search strategy raises `resume=True is only supported with a search_strategy`. Python/MSQ resume also requires a saved checkpoint directory.

## Read next

- Continue to [Log](Log.md) to understand the analysis surfaces built on top of UEL results.
- Continue to [Experiment Manifest](Experiment-Manifest.md) for manifest-driven SFD construction.
- Continue to [Trainer](Trainer.md) for reconstruction of saved YAML experiment rounds into sensors.

## Round storage format

New result directories from 5.31.1 declare `round_storage: gzip-jsonl-v2` in metadata and write `round_data.jsonl.gz`. Each gzip member contains one UTF-8 JSON round plus a newline, compressed at fixed level 1 with mtime zero and no filename header. Retained walk-forward fold objects receive gzip compression only; the [recording policy](#recording-policy) determines their saved fields. Existing directories without the discriminator retain plain `round_data.jsonl`; directories declaring `gzip-jsonl-v1` keep their original compressed representation, including on resume. Neither layout is migrated. Compression preserves the existing recording flags.

For ordinary rounds, a non-null top-level `market` is replaced in place by `market_ref`. Its lowercase SHA-256 digest addresses `round_market/<digest>.json.gz`. The decompressed blob bytes are exactly `json.dumps(market, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('utf-8')`; hash those bytes and verify them before reconstruction. Alignment stays in the owning round. Restore the market key at the reference's position; missing/null market stays missing/null. This preserves exact values, numeric types, signed zero and ordering without lossy conversion. Existing nonfinite tokens elsewhere in round/fold results retain the legacy JSON behavior.

In v2 ordinary rounds, retained top-level `preds` and `probs` flat finite numeric/boolean/null arrays are replaced at their original key position by `preds_ref` or `probs_ref`. Each reference contains `sha256` and the integer element `count`; its digest addresses `round_signals/<digest>.json.gz`. Blob bytes use the same canonical JSON encoding as market blobs. The reader verifies the hash, flat scalar types and declared count before restoring the array; references sharing a round must agree on row count and any retained execution positions. Integer/float/boolean distinctions, nulls and signed zero survive JSON reconstruction. Missing/null fields and nested or nonfinite values retain their inline representation. Nested walk-forward fold payloads are unchanged.

References are local to one run. They share array bytes, while each occurrence keeps its identity, typed parameters, alignment, threshold rule, metrics and execution evidence. All local evidence remains retained and the existing private reader resolves it for Trainer, Cohort, acceptance and resume. The recording policy still controls which groups are saved. There is no additional switch, sink, acknowledgement, receipt or retention policy.

External readers must verify each member trailer before exposing its round. The internal shared reader uses `zlib.decompressobj(wbits=31)`, requires each gzip header, rejects zero padding, and checks EOF/CRC/ISIZE. It validates references before constructing a path and verifies blob digest and return length against predictions, or retained execution positions when predictions are omitted. Decompression is bounded to 512 MiB per round and 128 MiB per blob. The package-local shared reader remains private; the project helper above handles only its declared v2 prediction payloads.

Trainer, Cohort and acceptance processing reject incomplete artifacts. Live/resume reads may ignore only an incomplete final member outside the successful checkpoint prefix. Resume validates ordered successful CSV identities, scans existing member/line offsets, and truncates in place: committed round-data bytes and inode survive. CSV retains its existing rewrite behavior. This is sequential restoration, not indexed random access.

Blob publication precedes its referencing member through a same-directory temporary file and atomic replace. The contract covers process interruption, not new power-loss/fsync durability or multiple writers in one directory. Orphan blobs and temporary leftovers are tolerated, never used as alternate references, and included in storage accounting; no garbage collector is introduced. Unknown/conflicting layouts and integrity failures raise errors before successful resume or report replacement.

Sharing benefits repeated arrays. Unique or small arrays can cost more, including filesystem allocation and reference overhead; compression and deduplication do not bound total run storage. Neither changes worker IPC, model execution or scientific results. Unique markets still require unique files and save less. Grid currently pins its Limen package version: update its reader before upgrading that separate dependency pin. No historical run is converted automatically.

### Recorded signal-sharing measurement

Reproduce the implemented-format comparison with `python -m pytest -q -s tests/test_round_signals.py -k recorded_storage_savings --log-cli-level=INFO`. It uses the existing recorded hourly fixture, fixed train/validation/test dates, deterministic single-threaded LightGBM parameters and a 16-round factorized sweep across four fee values. Accounting includes the round log, all market/signal blobs and reference metadata; resolved records must equal the v1 representation exactly. The test reports single-round costs as well as the repeated workload. These are small recorded-workload measurements, not campaign projections or a throughput promise.

Measured on 5.31.1 using fixture SHA-256 `a634bbd4` (prefix), the first 1,200 rows and 503 retained test rows. Splits are train Jan 1–20, validation Jan 20–25 and test Jan 25–Feb 15, 2024. The test pins lookback `[2, 4]`, LightGBM estimators `[16, 32]` and fees `[4, 9, 14, 19]`; the remaining deterministic model parameters and recording flags are in its `economic_sweep` fixture. Four distinct prediction/probability pairs recur across the fee values.

| Recorded workload | v1 bytes, all recording files | v2 bytes, all recording files | v1 / v2 allocated bytes |
|---|---:|---:|---:|
| One round | 16,960 | 16,852 | 24,576 / 32,768 |
| Four distinct pairs | 52,388 | 51,966 | 61,440 / 90,112 |
| Sixteen rounds, four repeated pairs | 193,534 | 131,557 | 200,704 / 167,936 |

The repeated case saves 61,977 bytes (32.0%), including format metadata and every market/signal blob. It uses eleven recording files rather than three. The production sweep took 0.354 seconds; the separate serialization samples took 0.016 seconds for v1 and 0.027 seconds for v2. These single local samples show write overhead, not a speedup. Allocated bytes depend on filesystem block size; unique/single cases can use more blocks despite slightly fewer serialized bytes. The regression asserts a relative saving on the repeated recorded case, not fixed compressed sizes or timing.

### Reuse of line computations

Each `run()` invocation owns a bounded cache for deterministic price-line
primitives and line-feature columns. It applies to ordinary sequential sweeps,
including random search, without enabling `factorize`. Keys contain the consumed
Float64 close values and the complete typed parameter set. Changed input order,
prices or carried context therefore require fresh computation whenever they
change the consumed values. The cache does not retain fitted scalers, models,
complete prepared frames or target labels. Returned arrays and series remain
isolated from cache storage. Entries are released when the invocation exits.

This reuse preserves the existing preparation, prediction, recording and search
contracts. Parallel processes retain their existing execution behavior.

Opt-in walk-forward `recording.fold_details` also retains versioned promotion witnesses. Missing clock, supported dtype, membership or fitted-model evidence records a visible unavailability reason without changing research admission. See [Trainer](Trainer.md#recorded-walk-forward-folds) and [qualified Cohort composition](Cohort.md#recorded-walk-forward-members).
