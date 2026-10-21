# Reducers and feedback

Reducers and feedback are optional controls inside the MSQ execution path shared by Python Research and Compiled Research. They let a run react to its own results instead of sweeping the original domain unchanged; using a search strategy does not require enabling them.

This layer is coordinated by `FeedbackController`, which can collect interventions from three sources:

- pruning strategies, also called reducers
- an in-process `intra_callback`
- an optional JSON intervention file

The [Python project](examples/python-research/README.md) constructs its own SFD and controller configuration. The [Compiled project](examples/compiled-research/README.md) executes its declared pipeline through the same controller. Their `search` scenarios include a budget-only run and a separate mixed feedback run.

## Prerequisites

- a CLI or [Python/MSQ run](Universal-Experiment-Loop.md#direct-python-run)
- a positive `feedback_interval`
- enough completed rounds for the selected reducer's observation threshold
- an `experiment_dir` when file-driven interventions or a saved `audit.jsonl` are needed

## Feedback cycle order

Each feedback trigger runs sources in this order:

1. pruning strategies
2. `intra_callback`
3. intervention file

The order matters because later sources see the `MSQ` after earlier sources have already modified it.

Two other behavior rules matter:

- failures are isolated per source, so one bad source does not block the others
- suggestion interventions are logged but not dispatched to the queue

## The intervention surface

The `MSQ` layer currently supports these intervention families:

| Operation | Effect |
|---|---|
| `remove_is` | remove one exact parameter value |
| `remove_ge`, `remove_le` | remove numeric values above or below a threshold |
| `keep_is`, `keep_between` | narrow a parameter to one value or a numeric range |
| `inject_value` | add a new value into the parameter domain |
| `trim` | reduce the remaining queue budget |
| `set_filter`, `clear_filter` | apply or remove named multi-combination filters |
| `remove_custom` | direct callback-only custom combo filtering |

Reducers return declarative dicts by default. The `intra_callback` receives direct `MSQ` access and can call queue methods itself.

## Built-in reducers

The project budget-only runs request four trials and stop at three completed rows. The proof checks the trim intervention, completed results and checkpoint budget in both interfaces:

<!-- example: feedback-python-budget source=docs/examples/python-research/scenarios.py region=adaptive_search scenario=search interface=python role=executed -->
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

<!-- example: feedback-compiled-budget source=docs/examples/compiled-research/adaptive.yaml region=budget_config scenario=search interface=compiled role=executed -->
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

These small runs exercise `BudgetReducer`. The four metric-driven alternatives below need enough observations for their respective analyses; a three-round example does not demonstrate their statistical reliability.

### `BudgetReducer`

BudgetReducer limits resource use by wall time or permutation count. Random trimming does not rank model quality; worst-first trimming requires a metric and removes its worst-scoring values before reducing the remaining budget.

<!-- reference: feedback-budget_reducer-constructor source=limen/experiment/reducer/budget_reducer.py region=signature:BudgetReducer.__init__ role=signature -->
```python
def __init__(self, *, start_time: float | None=None, max_walltime_hours: float | None=None, max_permutations: int | None=None, trim_strategy: str='random', check_after_pct: float=0.1, metric: str | None=None, maximize: bool=True, active: bool=True) -> None:
    ...
```

### `FocusReducer`

FocusReducer reacts to a breakthrough score. It uses named filters to narrow numeric and categorical choices around the winner, injects nearby numeric variations, and restores exploration after its timeout.

<!-- reference: feedback-focus_reducer-constructor source=limen/experiment/reducer/focus_reducer.py region=signature:FocusReducer.__init__ role=signature -->
```python
def __init__(self, *, metric: str, breakthrough_threshold: float, maximize: bool=True, focus_range_pct: float=0.2, focus_timeout: int=5, variation_count: int=5, min_observations: int=10, active: bool=True) -> None:
    ...
```

### `SaturationReducer`

SaturationReducer detects values with low metric variation. A named sampling filter retains a monitoring fraction and can be removed when variation returns.

<!-- reference: feedback-saturation_reducer-constructor source=limen/experiment/reducer/saturation_reducer.py region=signature:SaturationReducer.__init__ role=signature -->
```python
def __init__(self, *, metric: str, cv_threshold: float=0.01, window_size: int=100, min_samples_per_value: int=20, retain_fraction: float=0.1, active: bool=True) -> None:
    ...
```

### `SanityReducer`

SanityReducer removes parameter values whose metric has an excessive null/NaN rate. Its optional zero-metric, timeout and warning detectors emit suggestions rather than automatic removals.

<!-- reference: feedback-sanity_reducer-constructor source=limen/experiment/reducer/sanity_reducer.py region=signature:SanityReducer.__init__ role=signature -->
```python
def __init__(self, *, metric: str, nan_threshold: float=0.1, min_observations: int=1, zero_metric_threshold: float | None=None, execution_time_column: str='execution_time', execution_time_threshold: float | None=None, timeout_rate_threshold: float=0.5, warning_threshold: float | None=None, active: bool=True) -> None:
    ...
```

### `CorrelationReducer`

CorrelationReducer uses bootstrap parameter-to-metric correlation analysis to remove wrong-direction values. It can suggest keeping low-impact values; observation and sign-stability thresholds determine when it acts.

<!-- reference: feedback-correlation_reducer-constructor source=limen/experiment/reducer/correlation_reducer.py region=signature:CorrelationReducer.__init__ role=signature -->
```python
def __init__(self, *, metric: str, method: str='spearman', min_observations: int=50, prune_threshold: float=0.05, sign_stability_threshold: float=0.8, n_boot: int=300, negative_correlation_threshold: float=-0.3, maximize: bool=True, random_state: int=0, active: bool=True) -> None:
    ...
```

## When to use which reducer

The five reducers answer different questions. Pick by the problem you have, not by wanting "more feedback".

| Reducer | Reach for it when | What it does | Reversible |
|---|---|---|---|
| `BudgetReducer` | the sweep must finish inside a walltime or permutation cap | trims the remaining queue once, when the budget is projected to overrun | no — a trim is permanent |
| `SanityReducer` | you want to stop wasting rounds on broken parameter values | hard-removes values whose metric is mostly null; flags zero-metric, timeout, and warning-heavy values as suggestions | removals no; suggestions are advisory |
| `CorrelationReducer` | you want to prune values that move the metric the wrong way or barely at all | removes wrong-direction values, suggests keeps for low-impact ones | removals no; suggestions advisory |
| `FocusReducer` | a breakthrough score has appeared and you want to exploit near the winner | narrows each parameter around the best round and injects nearby variations | yes — named filters, snaps back after a timeout |
| `SaturationReducer` | a value's metric has stopped varying and no longer earns full budget | down-samples saturated values, keeping a monitoring sample | yes — restores a value if it starts varying again |

`SanityReducer` identifies unusable observations; its removals still change the search population. `BudgetReducer` is a resource cap. The other three change *what* the search explores, so they carry more risk of steering the run — introduce them one at a time and read `audit.jsonl` to see their effect.

Reducers combine, but with no conflict resolution. Within a feedback trigger they run in the order of the list they were constructed in, each analyzing the same pre-trigger `MSQ`; their interventions are dispatched afterwards with last-write-wins at the queue. `FocusReducer` and `SaturationReducer` use namespaced filter keys, so they do not clobber each other, but two exploitation reducers can still fight (e.g. a `BudgetReducer` trim shrinking the queue that a `FocusReducer` is injecting into). See [Feedback cycle order](#feedback-cycle-order) and [One concrete mixed feedback cycle](#one-concrete-mixed-feedback-cycle).

## Tuning guidelines

Every reducer takes `metric` (the column it reasons over) and `active` (default `True`). The knobs below are the ones that change behavior most; each reducer validates its ranges and raises `ValueError` on bad input.

`BudgetReducer` — a resource cap, not a quality judge:

| Parameter | Default | Effect |
|---|---|---|
| `max_walltime_hours` | `None` | wall-clock ceiling; the trim target is the smaller of the walltime and permutation projections |
| `max_permutations` | `None` | hard cap on completed rounds |
| `check_after_pct` | `0.1` | fraction of budget consumed before a trim is allowed; larger waits for a steadier throughput estimate, smaller reacts earlier on noisier estimates |
| `trim_strategy` | `'random'` | `'random'` downsamples blindly; `'worst_first'` removes the worst-scoring values first and requires `metric` |

`SanityReducer` — only the null-rate path removes; the rest suggest:

| Parameter | Default | Effect |
|---|---|---|
| `nan_threshold` | `0.1` | null/NaN rate above which a value is hard-removed; smaller is stricter |
| `min_observations` | `1` | trials required before a value can be removed, guarding against tiny samples |
| `zero_metric_threshold` / `execution_time_threshold` / `warning_threshold` | `None` | opt-in suggestion detectors; each stays off until set |

`CorrelationReducer` — needs enough rounds to be meaningful:

| Parameter | Default | Effect |
|---|---|---|
| `min_observations` | `50` | rounds required before it acts; larger is more conservative |
| `negative_correlation_threshold` | `-0.3` | wrong-direction trigger; nearer `0` prunes more readily, more negative prunes only strong offenders |
| `prune_threshold` | `0.05` | absolute correlation below this marks a value low-impact (a `keep_is` suggestion) |
| `sign_stability_threshold` | `0.8` | bootstrap sign agreement required before acting; larger demands higher confidence |

`FocusReducer` — exploitation around a breakthrough:

| Parameter | Default | Effect |
|---|---|---|
| `breakthrough_threshold` | required | metric level that activates focus; set near the top of the expected range so it only fires on a real breakthrough |
| `focus_range_pct` | `0.2` | half-width of the numeric window kept around the winner; larger explores wider, smaller searches tighter |
| `focus_timeout` | `5` | rounds without improvement before it snaps back to the full domain |
| `variation_count` | `5` | number of nearby values injected per numeric parameter |

`SaturationReducer` — variance-collapse detection:

| Parameter | Default | Effect |
|---|---|---|
| `cv_threshold` | `0.01` | coefficient of variation below which a value is saturated; larger declares saturation sooner |
| `window_size` | `100` | rolling tail length per value; larger is smoother and slower to react |
| `min_samples_per_value` | `20` | observations required before the CV is trusted |
| `retain_fraction` | `0.1` | share of a saturated value's combinations kept for monitoring; nearer `0` prunes harder |

## Feedback sources beyond reducers

### `intra_callback`

A Python Research callback receives `(log, msq)` and applies queue methods directly. Compiled Research declares reducers in YAML and can receive file interventions through the running CLI process.

Use a callback for caller-owned Python control logic. The Compiled example declares its resource limit with the budget reducer.

### `interventions.json`

With an `experiment_dir`, the controller watches that directory's `interventions.json`. The caller supplies a JSON array of declarative operations; Limen does not create this control file for the caller. The controller polls modification time and applies changed contents on the next feedback trigger. An unchanged file is not applied again in the same controller lifetime; controller checkpoint state retains the observed modification time.

The Python callback example writes its file during the first feedback cycle. An operator may instead write a Compiled run's intervention file while the CLI is running. A fresh experiment rejects a pre-existing control file; create it after execution begins.

## Audit trail

With an experiment directory, every feedback trigger writes one JSONL entry to `audit.jsonl`: round, applied interventions, suggestions, source errors and compact queue state. `checkpoint.json` retains the resulting domain and remaining budget. Read the completed rows as well: a domain mutation does not retroactively change observations already collected.

## One concrete mixed feedback cycle

Each project's separate mixed run enables all three feedback sources:

<!-- example: feedback-python-mixed source=docs/examples/python-research/scenarios.py region=mixed_feedback scenario=search interface=python role=executed -->
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

<!-- example: feedback-compiled-mixed source=docs/examples/compiled-research/adaptive.yaml region=budget_config scenario=search interface=compiled role=executed -->
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

The Python mixed-feedback example combines a reducer, callback and file injection. The Compiled YAML below demonstrates declared budget reduction; callbacks remain Python API extension behavior and are not part of the Compiled workflow.

Feedback changes the observations collected. This example establishes control ordering and persistence; it does not establish that narrowing or expanding the domain improves selection.

## Reducer registry

All built-in reducers are registered under the following keys:

| Key | Class |
|---|---|
| `budget` | `BudgetReducer` |
| `correlation` | `CorrelationReducer` |
| `focus` | `FocusReducer` |
| `sanity` | `SanityReducer` |
| `saturation` | `SaturationReducer` |

The public `REDUCER_REGISTRY` is available from `limen.experiment.reducer`. The current source definition is:

<!-- reference: feedback-registry source=limen/experiment/reducer/registry.py region=definition:REDUCER_REGISTRY role=definition -->
```python
REDUCER_REGISTRY: dict[str, type[PruningStrategy]] = {'budget': BudgetReducer, 'correlation': CorrelationReducer, 'focus': FocusReducer, 'sanity': SanityReducer, 'saturation': SaturationReducer}
```

## Configuring reducers in YAML

The Compiled project declares its budget reducer with the same registry key and constructor arguments:

<!-- example: feedback-compiled-budget-declaration source=docs/examples/compiled-research/adaptive.yaml region=budget_config scenario=search interface=compiled role=executed -->
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

`feedback_interval` schedules the feedback cycles. Each reducer `type` is a registry key; `params` supplies its constructor arguments. `limen validate` rejects unknown types and fields before execution. File interventions are read by the same controller during CLI execution.

## Read next

Continue to [search and recovery](Advanced-Search.md) for candidate generation and checkpointing, [the experiment loop](Universal-Experiment-Loop.md) for feedback scheduling, and [Trainer](Trainer.md) for reconstructible Compiled experiments.
