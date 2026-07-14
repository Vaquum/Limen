# Pruning strategies

Pruning strategies, also called reducers, analyze completed rounds during an advanced search and return declarative interventions. `FeedbackController` applies those interventions to the mutable search queue (`MSQ`), not to the reducer itself.

Use this page when implementing or reviewing a reducer. For user-facing configuration and behavior, start with [Reducers and Feedback](../Reducers-And-Feedback.md).

## Prerequisites

- an artifact-backed advanced run with a `SearchStrategy`
- a positive `feedback_interval`
- enough completed rounds for the selected reducer's minimum-observation rule
- a target metric column for reducers that analyze quality

## Reducer contract

Every reducer subclasses `PruningStrategy` and implements three methods:

```python
class PruningStrategy:
    def analyze_and_intervene(self, log, msq) -> list[dict]: ...
    def get_state(self) -> dict: ...
    def set_state(self, state: dict) -> None: ...
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

```python
BudgetReducer(
    start_time=None,
    max_walltime_hours=None,
    max_permutations=None,
    trim_strategy='random',
    check_after_pct=0.1,
    metric=None,
    maximize=True,
    active=True,
)
```

`trim_strategy` is `random` or `worst_first`. The latter requires `metric`. The reducer emits a final `trim` intervention even when it also removes values first, and it fires at most once.

### `CorrelationReducer`

```python
CorrelationReducer(
    metric='auc',
    method='spearman',
    min_observations=50,
    prune_threshold=0.05,
    sign_stability_threshold=0.8,
    n_boot=300,
    negative_correlation_threshold=-0.3,
    maximize=True,
    random_state=0,
    active=True,
)
```

Supported methods are `spearman`, `pearson`, and `kendall`. Non-numeric, all-null, and constant parameter columns are skipped.

### `FocusReducer`

```python
FocusReducer(
    metric='auc',
    breakthrough_threshold=0.8,
    maximize=True,
    focus_range_pct=0.2,
    focus_timeout=5,
    variation_count=5,
    min_observations=10,
    active=True,
)
```

On activation it emits `set_filter` and `inject_value` interventions around the best combination. After `focus_timeout` rounds without improvement it emits `clear_filter` and returns to the unfocused domain.

### `SanityReducer`

```python
SanityReducer(
    metric='auc',
    nan_threshold=0.1,
    min_observations=1,
    zero_metric_threshold=None,
    execution_time_column='execution_time',
    execution_time_threshold=None,
    timeout_rate_threshold=0.5,
    warning_threshold=None,
    active=True,
)
```

Only the null-rate path dispatches removals. Zero-metric, slow-run, and warning-rate checks are optional suggestions: they are written to the audit trail but not applied to `MSQ`.

### `SaturationReducer`

```python
SaturationReducer(
    metric='auc',
    cv_threshold=0.01,
    window_size=100,
    min_samples_per_value=20,
    retain_fraction=0.1,
    active=True,
)
```

The reducer measures recent coefficient of variation per parameter value. It emits `set_filter` when a value saturates and `clear_filter` if variation later recovers.

## YAML configuration

The CLI compiles `uel.pruning_strategies` through `REDUCER_REGISTRY` and passes the instances to both new and resumed runs:

```yaml
uel:
  n_permutations: 200
  feedback_interval: 20
  pruning_strategies:
    - type: sanity
      params:
        metric: auc
        min_observations: 10
    - type: budget
      params:
        max_permutations: 100
        check_after_pct: 0.25
```

Each entry must be a mapping with a registered `type`. `params` is optional but, when present, must be a mapping of constructor arguments. Unknown reducer keys, malformed entries, and invalid constructor values fail validation or compilation before the run starts.

## Python configuration

```python
from limen.experiment.reducer import BudgetReducer, SanityReducer
from limen import UniversalExperimentLoop

reducers = [
    SanityReducer(metric='auc', min_observations=10),
    BudgetReducer(max_permutations=100),
]

uel = UniversalExperimentLoop(
    sfd=my_sfd,
    experiment_dir='results/example',
    search_strategy=strategy,
    feedback_interval=20,
    pruning_strategies=reducers,
)
```

Reducers run in list order against the same pre-trigger log and queue state. Their returned interventions are dispatched in that order; conflicting changes therefore require deliberate composition.

## Manual intervention

Two mechanisms exist; neither is a reducer class.

### In-process callback

`intra_callback(log, msq)` receives direct `MSQ` access and mutates it itself. The controller records new `msq.intervention_log` entries after the callback returns. Callback failures are isolated and logged.

```python
def manual_callback(log, msq):
    if log.height >= 20:
        msq.remove_is('learning_rate', 1.0)
```

### Intervention file

Artifact-backed runs poll `interventions.json` in the experiment directory. When its modification time changes, the controller expects a JSON array of ordinary intervention dictionaries. Missing `source` fields are filled with `intervention_file`.

```json
[
  {"op": "remove_is", "param": "learning_rate", "value": 1.0},
  {"op": "trim", "target_count": 50}
]
```

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

```bash
python -m pytest -q tests/test_*_reducer.py tests/test_reducer_factory.py tests/test_yaml.py
```

## Read next

- [Reducers and Feedback](../Reducers-And-Feedback.md) for operator behavior and tuning
- [Advanced Search](../Advanced-Search.md) for the artifact-backed execution path
- [Universal Experiment Loop](../Universal-Experiment-Loop.md) for scheduling and persistence
