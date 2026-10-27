# Pruning strategies

Pruning strategies, also called reducers, analyze completed rounds in a CLI or Python/MSQ run and return declarative interventions. `FeedbackController` applies those interventions to the mutable search queue (`MSQ`), not to the reducer itself.

Use this page when implementing or reviewing a reducer. For user-facing configuration and behavior, start with [Reducers and Feedback](../Reducers-And-Feedback.md).

The [Python project](../examples/python-research/README.md) and
[Compiled project](../examples/compiled-research/README.md) both exercise a
three-candidate budget. The Python project also exercises mixed reducer, callback
and file feedback. Maintain its ordering, injected-candidate scores and checkpoint
assertions when changing the controller or queue. The full functions below use their projects'
imports and recorded-data helpers.

## Prerequisites

- a CLI or Python/MSQ run with a `SearchStrategy`; persistence is needed only for saved checkpoints or audit entries
- a positive `feedback_interval`
- enough completed rounds for the selected reducer's minimum-observation rule
- a target metric column for reducers that analyze quality

## Reducer contract

Every reducer subclasses `PruningStrategy` and implements three methods:

<!-- reference: pruning-analyze_and_intervene source=limen/experiment/reducer/pruning_strategy.py region=signature:PruningStrategy.analyze_and_intervene role=signature -->
```python
@abstractmethod
def analyze_and_intervene(self, log: Any, msq: Any) -> list[dict[str, Any]]:
    ...
```

<!-- reference: pruning-get_state source=limen/experiment/reducer/pruning_strategy.py region=signature:PruningStrategy.get_state role=signature -->
```python
@abstractmethod
def get_state(self) -> dict[str, Any]:
    ...
```

<!-- reference: pruning-set_state source=limen/experiment/reducer/pruning_strategy.py region=signature:PruningStrategy.set_state role=signature -->
```python
@abstractmethod
def set_state(self, state: dict[str, Any]) -> None:
    ...
```

`analyze_and_intervene()` may inspect `msq` for observability but must not mutate it. It returns zero or more intervention dictionaries. `get_state()` and `set_state()` make reducer state resumable through checkpoints. Setting `active=False` makes a reducer return no interventions.

## Registered reducers

`REDUCER_REGISTRY` contains exactly five keys:

| Key | Class | Required arguments | Effect |
| --- | --- | --- | --- |
| `budget` | `BudgetReducer` | one of `max_walltime_hours`, `max_permutations` | trims once when the projected run exceeds budget |
| `correlation` | `CorrelationReducer` | `metric` | removes wrong-direction numeric values and can log low-impact suggestions |
| `focus` | `FocusReducer` | `metric`, `breakthrough_threshold` | installs temporary filters around a breakthrough and injects nearby numeric values |
| `sanity` | `SanityReducer` | `metric` | removes values with excessive null metrics and can log advisory anomaly suggestions |
| `saturation` | `SaturationReducer` | `metric` | samples saturated parameter values through reversible named filters |

There is no `ManualReducer`. Manual control uses the callback or intervention-file surfaces described below.

## Constructor reference

### `BudgetReducer`

<!-- reference: pruning-budget source=limen/experiment/reducer/budget_reducer.py region=signature:BudgetReducer.__init__ role=signature -->
```python
def __init__(self, *, start_time: float | None=None, max_walltime_hours: float | None=None, max_permutations: int | None=None, trim_strategy: str='random', check_after_pct: float=0.1, metric: str | None=None, maximize: bool=True, active: bool=True) -> None:
    ...
```

`trim_strategy` is `random` or `worst_first`. The latter requires `metric`. The reducer emits a final `trim` intervention even when it also removes values first, and it fires at most once.

### `CorrelationReducer`

<!-- reference: pruning-correlation source=limen/experiment/reducer/correlation_reducer.py region=signature:CorrelationReducer.__init__ role=signature -->
```python
def __init__(self, *, metric: str, method: str='spearman', min_observations: int=50, prune_threshold: float=0.05, sign_stability_threshold: float=0.8, n_boot: int=300, negative_correlation_threshold: float=-0.3, maximize: bool=True, random_state: int=0, analyzer: str='correlation', metrics: Mapping[str, bool] | None=None, n_repeats: int=10, min_model_score: float=0.1, active: bool=True) -> None:
    ...
```

Supported methods are `spearman`, `pearson`, and `kendall`. Non-numeric, all-null, and constant parameter columns are skipped.

Set `analyzer="random_forest"` to use the [held-out parameter importance analysis](../Log.md) instead. It handles numeric, Boolean and categorical scalar parameters together, including nonlinear effects and interactions. It requires at least 20 distinct successful configurations and `min_observations` successful rows. Each metric's held-out surrogate R² must meet `min_model_score`; otherwise it logs the unavailable evidence and emits no suggestions. Missing columns, nonfinite successful metrics and unsupported parameter values raise errors.

Forest analysis emits advisory `keep_is` suggestions when permutation importance mean (floored at zero) plus twice its repeat standard deviation is below `prune_threshold`. This is a screening rule, not a confidence interval. `n_repeats` controls repeated shuffles; `random_state` controls the held-out configuration split, forest and shuffles. Identical configurations stay together across that split. The forest does not infer a monotonic wrong direction.

For multiple objectives, `metrics` maps additional metric names to explicit maximize booleans. The primary `metric` uses `maximize`. Every metric must pass the surrogate-score gate and low-importance rule, and agree on the same observed best value before a suggestion is retained. Conflicting directions for the primary metric are rejected. Suggestions retain native queue values and are checkpointed through the existing `suggested` state. YAML uses the same controls under `type: correlation` → `params`.

### `FocusReducer`

<!-- reference: pruning-focus source=limen/experiment/reducer/focus_reducer.py region=signature:FocusReducer.__init__ role=signature -->
```python
def __init__(self, *, metric: str, breakthrough_threshold: float, maximize: bool=True, focus_range_pct: float=0.2, focus_timeout: int=5, variation_count: int=5, min_observations: int=10, active: bool=True) -> None:
    ...
```

On activation it emits `set_filter` and `inject_value` interventions around the best combination. After `focus_timeout` rounds without improvement it emits `clear_filter` and returns to the unfocused domain.

### `SanityReducer`

<!-- reference: pruning-sanity source=limen/experiment/reducer/sanity_reducer.py region=signature:SanityReducer.__init__ role=signature -->
```python
def __init__(self, *, metric: str, nan_threshold: float=0.1, min_observations: int=1, zero_metric_threshold: float | None=None, execution_time_column: str='execution_time', execution_time_threshold: float | None=None, timeout_rate_threshold: float=0.5, warning_threshold: float | None=None, active: bool=True) -> None:
    ...
```

Only the null-rate path dispatches removals. Zero-metric, slow-run, and warning-rate checks are optional suggestions: they are written to the audit trail but not applied to `MSQ`.

### `SaturationReducer`

<!-- reference: pruning-saturation source=limen/experiment/reducer/saturation_reducer.py region=signature:SaturationReducer.__init__ role=signature -->
```python
def __init__(self, *, metric: str, cv_threshold: float=0.01, window_size: int=100, min_samples_per_value: int=20, retain_fraction: float=0.1, active: bool=True) -> None:
    ...
```

The reducer measures recent coefficient of variation per parameter value. It emits `set_filter` when a value saturates and `clear_filter` if variation later recovers.

## YAML configuration

The CLI compiles `uel.pruning_strategies` through `REDUCER_REGISTRY`. This is the Compiled project's actual four-candidate declaration with a three-candidate budget:

<!-- example: pruning-compiled-definition source=docs/examples/compiled-research/adaptive.yaml region=__file__ scenario=search interface=compiled role=executed -->
```yaml
# region declaration
schema_version: '1.0'
metadata:
  name: book_compiled_adaptive
  mode: development
sfd:
  manifest:
    type: ml
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
    scaler:
      from_params: scaler_type
    reference_architecture: limen.sfd.reference_architecture.logreg_binary
  params:
    scaler_type:
    - logreg
    C:
    - 0.1
    - 0.5
    - 1.0
    - 2.0
    random_state:
    - 42
# region budget_config
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
# endregion budget_config
# endregion declaration
```

Each entry must be a mapping with a registered `type`. `params` is optional but, when present, must be a mapping of constructor arguments. Unknown reducer keys, malformed entries, and invalid constructor values fail validation or compilation before the run starts.

## Python configuration

The Python project supplies its own SFD and puts the same budget on its queue:

<!-- example: pruning-python-budget source=docs/examples/python-research/scenarios.py region=adaptive_search scenario=search interface=python role=executed -->
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

Reducers run in list order against the same pre-trigger log and queue state. Their returned interventions are dispatched in that order; conflicting changes therefore require deliberate composition.

## Manual intervention

Two mechanisms exist; neither is a reducer class.

### In-process callback

`intra_callback(log, msq)` receives direct `MSQ` access and mutates it itself. The controller records new `msq.intervention_log` entries after the callback returns. Handled callback failures are isolated, warned about, and recorded in the audit `errors` list; partial intervention-log entries are retained. The Python project keeps one parameter value, then writes the intervention file during that same feedback cycle:

<!-- example: pruning-python-callback source=docs/examples/python-research/scenarios.py region=mixed_feedback scenario=search interface=python role=executed -->
```python
def mixed_feedback(data_dir, output):
    space = {'window': [4, 12], 'alpha': [.1, 1.]}

    def steer(log, msq):
        if log.height == 2:
            msq.keep_is('alpha', .1)
            (output / 'interventions.json').write_text(json.dumps([
                {'op': 'inject_value', 'param': 'alpha', 'value': 2.0}]))

    def model(data, round_params):
        fitted = project.fit(data['train'], round_params)
        return {'validation_net_return': project.backtest(data['validation'], fitted.predict(data['validation'].select('momentum')))}

    sfd = SimpleNamespace(__name__=__name__, params=lambda: space, prep=recordable_prep, model=model)
    loop = UniversalExperimentLoop(data=hourly(data_dir, 1440), sfd=sfd,
                                   search_strategy=GridStrategy(ParamDomain(space)), experiment_dir=output,
                                   checkpoint_interval=1, feedback_interval=1, intra_callback=steer,
                                   pruning_strategies=[BudgetReducer(max_permutations=3)])
    loop.run('python-feedback', n_permutations=4, workers=1, progress_bar=False)
    checkpoint = json.loads((output / 'checkpoint.json').read_text())
    return {'rows': project.completed_log(loop).select('window', 'alpha', 'validation_net_return').to_dicts(),
            'audit': [json.loads(line) for line in (output / 'audit.jsonl').read_text().splitlines()],
            'domain': checkpoint['domain_state'], 'yielded': checkpoint['msq_state']['yielded_count'],
            'remaining': checkpoint['msq_state']['trim_budget']}
```

### Intervention file

Artifact-backed controller integrations poll `interventions.json` in the experiment directory. When its modification time changes, the controller expects a JSON array of ordinary intervention dictionaries. Missing `source` fields are filled with `intervention_file`. The Python mixed-feedback example above creates the file during execution; precreating it in a fresh results directory fails the initial-state guard. An unchanged file is not applied twice within the same controller lifetime, and checkpoints retain its observed modification time.

This is an optional component integration surface. The canonical Compiled project keeps its research controls in YAML through `uel.pruning_strategies`.

## Intervention operations

The dispatcher supports:

- `remove_is`, `remove_ge`, `remove_le`
- `keep_is`, `keep_between`
- `inject`, `inject_value`
- `trim`
- `set_filter`, `clear_filter`

An intervention with `action: suggest` is audit-only. It is not dispatched and is excluded from the applied-intervention list passed to `SearchStrategy.update_from_feedback()`.

## Feedback-cycle order

At each trigger, the controller processes:

1. registered pruning strategies
2. the in-process callback
3. the intervention file

Each source is failure-isolated. Applied interventions, suggestions, errors, and the resulting queue state are written to `audit.jsonl` when an audit path is configured.

## Verification

Reducer changes must prove:

- constructor boundary validation
- emitted intervention dictionaries
- `MSQ` dispatch behavior
- suggestion non-dispatch
- checkpoint state round-trip
- source failure isolation
- YAML registry and compiler wiring when the public surface changes

Run the focused coverage with:

The focused repository command is `python -m pytest -q tests/test_*_reducer.py tests/test_reducer_factory.py tests/test_yaml.py`. These reducer/unit tests complement the two recorded-data project cases; a constructor reference does not claim that all five strategies ran in the small project.

## Read next

- [Reducers and Feedback](../Reducers-And-Feedback.md) for operator behavior and tuning
- [Advanced Search](../Advanced-Search.md) for shared search and optional persistence
- [Universal Experiment Loop](../Universal-Experiment-Loop.md) for scheduling and persistence
